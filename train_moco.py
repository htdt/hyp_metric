import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from tqdm import trange
import wandb
import copy
from collections import deque
import timm
import multiprocessing
from functools import partial

from utils import eval_model, freeze_model, NormLayer
from proxy_anchor.dataset import CUBirds, SOP
import hyptorch.nn as hypnn
from hyptorch.pmath import dist_matrix
from cfg import get_cfg


def ss_loss(x, x_t, y, y_t, temperature):
    """
    x - batch of embeddings,
    x_t - target embeddings,
    y - labels for x,
    y_t - labels for x_t,
    temperature - logits coef,
    based on https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    """

    mask = torch.eq(y.view(-1, 1), y_t.view(1, -1)).float().cuda()

    anchor_dot_contrast = x @ x_t.T / temperature
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
    loss = -mean_log_prob_pos.mean()
    stats = {
        "logits/min": logits.min().item(),
        "logits/mean": logits.mean().item(),
    }
    return loss, stats


def hyp_loss(x, x_t, y, y_t, hyp_c, temperature):
    """
    x - batch of embeddings,
    x_t - target embeddings,
    y - labels for x,
    y_t - labels for x_t,
    hyp_c - hyperbolic curvature,
    temperature - logits coef
    """

    mask = torch.eq(y.view(-1, 1), y_t.view(1, -1)).float().cuda()

    dm = torch.empty(len(x), len(x_t), device="cuda")
    for i in range(len(x)):
        dm[i : i + 1] = dist_matrix(x[i : i + 1], x_t, hyp_c)
    dm = -dm / temperature
    # dm = -dist_matrix(x, x_t, hyp_c) / temperature
    logits_max, _ = torch.max(dm, dim=1, keepdim=True)
    logits = dm - logits_max.detach()

    log_prob = logits - torch.log(logits.exp().sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
    loss = -mean_log_prob_pos.mean()
    stats = {
        "logits/min": logits.min().item(),
        "logits/mean": logits.mean().item(),
    }
    return loss, stats


if __name__ == "__main__":
    cfg = get_cfg()
    wandb.init(project="hyp_metric", config=cfg)

    train_tr = T.Compose(
        [
            T.RandomResizedCrop(
                224, scale=(0.2, 1.0), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds_f = {"CUB": CUBirds, "SOP": SOP}
    ds_train = ds_f[cfg.ds](cfg.path, "train", train_tr)
    dl_train = DataLoader(
        dataset=ds_train,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=False,
        drop_last=True,
    )

    model = timm.create_model(cfg.model, pretrained=True)
    if cfg.hyp_c > 0:
        last = hypnn.ToPoincare(c=cfg.hyp_c, ball_dim=cfg.emb, riemannian=False)
    else:
        last = NormLayer()
    model.head = nn.Sequential(nn.Linear(model.head.in_features, cfg.emb), last)
    if cfg.freeze is not None:
        freeze_model(model, cfg.freeze)
    model = nn.DataParallel(model)
    model.cuda().train()

    model_t = copy.deepcopy(model)
    model_t.cuda().train()
    for param in model_t.parameters():
        param.requires_grad = False

    def update_target(tau):
        for t, s in zip(model_t.parameters(), model.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))

    update_target(0)
    z_queue = deque(maxlen=cfg.qlen)
    y_queue = deque(maxlen=cfg.qlen)

    if cfg.hyp_c == 0:
        loss_f = partial(ss_loss, temperature=cfg.t)
    else:
        loss_f = partial(hyp_loss, temperature=cfg.t, hyp_c=cfg.hyp_c)
    eval_f = partial(eval_model, ds_name=cfg.ds, hyp_c=cfg.hyp_c)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    cudnn.benchmark = True
    for ep in trange(cfg.ep):
        loss_ep = []
        stats_ep = []
        for x, y in dl_train:
            x, y = x.cuda(), y.cuda()

            if len(z_queue):
                z = model(x)
                zz, yy = torch.cat(tuple(z_queue)), torch.cat(tuple(y_queue))
                loss, stats = loss_f(z, zz, y, yy)

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 3)
                optimizer.step()
                loss_ep.append(loss.item())
                stats_ep.append(stats)
                update_target(cfg.tau)

            with torch.no_grad():
                z_t = model_t(x)
            z_queue.append(z_t)
            y_queue.append(y)

        if (ep + 1) % cfg.eval_ep == 0:
            result = {
                "recall": eval_f(model),
                "recall_t": eval_f(model_t),
            }
            wandb.log(result, commit=False)
        stats = {
            k: sum(a[k] for a in stats_ep) / len(stats_ep) for k in stats_ep[0].keys()
        }
        wandb.log({"loss": sum(loss_ep) / len(loss_ep), "ep": ep, **stats})

    model.module.head = NormLayer()
    model_t.module.head = NormLayer()
    result = {
        "recall_b": eval_model(model, cfg.ds),
        "recall_bt": eval_model(model_t, cfg.ds),
    }
    wandb.log(result)
