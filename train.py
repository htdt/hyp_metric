import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from tqdm import trange
import wandb
import timm
import multiprocessing
from functools import partial
import pickle
import numpy as np
import os

from sampler import MPerClassSampler
from utils import eval_model, freeze_model, NormLayer
from proxy_anchor.dataset import CUBirds, SOP
import hyptorch.nn as hypnn
from hyptorch.pmath import dist_matrix
from cfg import get_cfg


def contrastive_loss(x0, x1, tau, hyp_c):
    if hyp_c == 0:
        dist_f = lambda x, y: x @ y.t()
    else:
        dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
    bsize = x0.shape[0]
    target = torch.arange(bsize).cuda()
    eye_mask = torch.eye(bsize).cuda() * 1e9
    logits00 = dist_f(x0, x0) / tau - eye_mask
    logits01 = dist_f(x0, x1) / tau
    logits = torch.cat([logits01, logits00], dim=1)
    logits -= logits.max(1, keepdim=True)[0].detach()
    loss = F.cross_entropy(logits, target)
    stats = {
        "logits/min": logits01.min().item(),
        "logits/mean": logits01.mean().item(),
        "logits/max": logits01.max().item(),
        "logits/acc": (logits01.argmax(-1) == target).float().mean().item(),
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
    lbl_fname = cfg.path + "/" + cfg.ds + "_labels.pickle"
    if os.path.isfile(lbl_fname):
        lbl = pickle.load(open(lbl_fname, "rb"))
    else:
        lbl = [x[1] for x in ds_train]
        pickle.dump(lbl, open(lbl_fname, "wb"))
    sampler = MPerClassSampler(lbl, 2, cfg.bs, len(ds_train))
    dl_train = DataLoader(
        dataset=ds_train,
        sampler=sampler,
        batch_size=cfg.bs,
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

    loss_f = partial(contrastive_loss, tau=cfg.t, hyp_c=cfg.hyp_c)
    eval_f = partial(eval_model, ds_name=cfg.ds, path=cfg.path, hyp_c=cfg.hyp_c)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    cudnn.benchmark = True
    for ep in trange(cfg.ep):
        stats_ep = []
        for x, y in dl_train:
            y = y.view(cfg.bs // 2, 2)
            assert (y[:, 0] == y[:, 1]).all()
            x = x.cuda()
            z = model(x).view(cfg.bs // 2, 2, cfg.emb)
            loss0, stats0 = loss_f(z[:, 0], z[:, 1])
            loss1, stats1 = loss_f(z[:, 1], z[:, 0])

            optimizer.zero_grad()
            (loss0 + loss1).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            stats_ep.append({**stats0, "loss": loss0.item()})
            stats_ep.append({**stats1, "loss": loss1.item()})

        if (ep + 1) % cfg.eval_ep == 0:
            wandb.log({"recall": eval_f(model)}, commit=False)
        stats_ep = {k: np.mean([x[k] for x in stats_ep]) for k in stats_ep[0]}
        wandb.log({**stats_ep, "ep": ep})

    model.module.head = NormLayer()
    wandb.log({"recall_b": eval_model(model, cfg.ds, cfg.path)})
