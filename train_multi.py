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

from utils import eval_model, ss_loss, freeze_model, NormLayer, MultiSample
from proxy_anchor.dataset import CUBirds, SOP
import hyptorch.nn as hypnn
from cfg import get_cfg
from hyp_loss import hyp_loss


if __name__ == "__main__":
    cfg = get_cfg()
    wandb.init(project="hyp_metric", config=cfg)

    train_tr = T.Compose(
        [
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
            T.RandomGrayscale(0.1),
            T.RandomResizedCrop(224, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if cfg.ds == "CUB":
        ds_train = CUBirds("/home/i", "train", MultiSample(train_tr))
    elif cfg.ds == "SOP":
        ds_train = SOP("/home/i", "train", MultiSample(train_tr))
    else:
        raise "dataset not found"

    dl_train = DataLoader(
        dataset=ds_train,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=False,
        drop_last=True,
    )

    model = timm.create_model(cfg.model, pretrained=True)
    if cfg.hyp:
        last = hypnn.ToPoincare(c=cfg.hyp_c, ball_dim=cfg.emb, riemannian=False)
    else:
        last = NormLayer()
    model.head = nn.Sequential(nn.Linear(384, cfg.emb), last)
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

    if not cfg.hyp:
        loss_f = partial(ss_loss, temperature=cfg.t)
    else:
        loss_f = partial(hyp_loss, temperature=cfg.t, hyp_c=cfg.hyp_c)
    eval_f = partial(eval_model, ds_name=cfg.ds, hyp_c=cfg.hyp_c if cfg.hyp else None)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    cudnn.benchmark = True
    for ep in trange(cfg.ep):
        loss_ep = []
        stats_ep = []
        for (x0, x1), y in dl_train:
            x0, x1, y = x0.cuda(), x1.cuda(), y.cuda()

            with torch.no_grad():
                z_t = model_t(x1)
            z_queue.append(z_t)
            y_queue.append(y)

            z = model(x0)
            zz, yy = torch.cat(tuple(z_queue)), torch.cat(tuple(y_queue))
            loss, stats = loss_f(z, zz, y, yy)

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            loss_ep.append(loss.item())
            stats_ep.append(stats)
            update_target(cfg.tau)

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

    model.module.head = nn.Identity()
    model_t.module.head = nn.Identity()
    result = {
        "recall_b": eval_model(model, cfg.ds),
        "recall_bt": eval_model(model_t, cfg.ds),
    }
    wandb.log(result)
