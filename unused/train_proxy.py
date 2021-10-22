import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from apex import amp
from apex.parallel import DistributedDataParallel

from functools import partial
from tqdm import trange
import wandb
import multiprocessing
import numpy as np
import PIL
from tap import Tap
from typing_extensions import Literal
import os

from helpers import get_emb, get_recall
from proxy_anchor.dataset import CUBirds, SOP
import hyptorch.nn as hypnn
from hyptorch.pmath import dist_matrix
from model import init_model


class Config(Tap):
    path: str = "/home/i"
    ds: Literal["SOP", "CUB"] = "CUB"
    bs: int = 700  # per GPU
    lr: float = 1e-5
    emb: int = 128
    freeze: int = 0
    ep: int = 100
    hyp_c: float = 0.1
    eval_ep: int = 10
    model: str = "dino_vits16"
    clip_grad: bool = False
    local_rank: int = 0


# https://github.com/dichotomies/proxy-nca/blob/master/proxynca.py
class ProxyNCA(nn.Module):
    def __init__(self, nb_classes, emb, hyp_c):
        super(ProxyNCA, self).__init__()
        p = torch.empty(nb_classes, emb)
        nn.init.orthogonal_(p)
        self.proxies = nn.Parameter(p)
        self.hyp_c = hyp_c
        if hyp_c > 0:
            self.tp = hypnn.ToPoincare(c=hyp_c, ball_dim=emb, riemannian=False)

    def forward(self, x, y):
        if self.hyp_c > 0:
            P = self.tp(self.proxies)
            D = dist_matrix(x, P, c=self.hyp_c) * 3
        else:
            P = F.normalize(self.proxies, p=2, dim=-1) * 3
            D = torch.cdist(x, P) ** 2
        T = nn.functional.one_hot(y, len(self.proxies)).float()
        T *= 0.9
        T[T == 0] = 0.1 / (len(self.proxies) - 1)
        loss = torch.sum(-T * F.log_softmax(-D, -1), -1).mean()
        stats = {
            "logits/acc": (D.argmin(-1) == y).float().mean().item(),
        }
        return loss, stats


if __name__ == "__main__":
    cfg: Config = Config().parse_args()
    if cfg.local_rank == 0:
        wandb.init(project="hyp_metric", config=cfg.as_dict())

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        world_size = torch.distributed.get_world_size()

    if cfg.model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tr = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=PIL.Image.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(*mean_std),
        ]
    )

    ds_f = {"CUB": CUBirds, "SOP": SOP}
    ds_train = ds_f[cfg.ds](cfg.path, "train", train_tr)
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(ds_train)
    else:
        sampler = None
    dl_train = DataLoader(
        dataset=ds_train,
        sampler=sampler,
        batch_size=cfg.bs,
        num_workers=multiprocessing.cpu_count() // world_size,
        pin_memory=True,
        drop_last=True,
        shuffle=sampler is None
    )

    model = init_model(cfg)
    criterion = ProxyNCA(100, cfg.emb, cfg.hyp_c)
    criterion.cuda().train()
    param = list(model.parameters()) + list(criterion.parameters())
    optimizer = optim.AdamW(param, lr=cfg.lr)
    [model, criterion], optimizer = amp.initialize(
        [model, criterion], optimizer, opt_level="O2", loss_scale=16.0
    )
    if world_size > 1:
        model = DistributedDataParallel(model, delay_allreduce=True)
    eval_f = partial(
        get_emb,
        model=model,
        ds=ds_f[cfg.ds],
        path=cfg.path,
        mean_std=mean_std,
        world_size=world_size,
    )

    cudnn.benchmark = True
    for ep in trange(cfg.ep):
        if sampler is not None:
            sampler.set_epoch(ep)
        stats_ep = []
        for x, y in dl_train:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            loss, stats = criterion(model(x), y)
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if cfg.clip_grad:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 3)
            optimizer.step()
            stats_ep.append({**stats, "loss": loss.item()})

        if (ep + 1) % cfg.eval_ep == 0:
            eval_head = eval_f()
            eval_body = eval_f(skip_head=True)

        if cfg.local_rank == 0:
            stats_ep = {k: np.mean([x[k] for x in stats_ep]) for k in stats_ep[0]}
            if (ep + 1) % cfg.eval_ep == 0:
                recall_head = get_recall(*eval_head, cfg.ds, cfg.hyp_c)
                recall_body = get_recall(*eval_body, cfg.ds, 0)
                stats_ep = {"recall": recall_head, "recall_b": recall_body, **stats_ep}
            wandb.log({**stats_ep, "ep": ep})
