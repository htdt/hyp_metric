import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from apex import amp
from apex.parallel import DistributedDataParallel

from tqdm import trange
import wandb
import timm
import multiprocessing
from functools import partial
import numpy as np
import PIL
from tap import Tap
from typing_extensions import Literal
import copy
from collections import deque

from helpers import get_emb, get_recall, freeze_model, NormLayer, HeadSwitch
from proxy_anchor.dataset import CUBirds, SOP
import hyptorch.nn as hypnn
from hyptorch.pmath import dist_matrix


class Config(Tap):
    path: str = "/home/i"
    ds: Literal["SOP", "CUB"] = "SOP"
    num_samples: int = 2
    bs: int = 400  # per GPU
    lr: float = 1e-5
    t: float = 0.3
    tau: float = 0.99
    qlen: int = 16
    emb: int = 128
    freeze: int = None
    ep: int = 10
    hyp_c: float = 0.1
    eval_ep: int = 10
    model: str = "dino_vits16"
    save_emb: bool = False
    emb_name: str = "emb"
    clip_grad: bool = False
    local_rank: int = 0


def get_loss(x, x_t, y, y_t, hyp_c, temperature):
    mask = torch.eq(y.view(-1, 1), y_t.view(1, -1)).float().cuda()
    if hyp_c == 0:
        dist_f = lambda x, y: x @ y.t()
    else:
        dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
    logits = torch.empty(len(x), len(x_t), dtype=x.dtype, device="cuda")
    for i in range(len(x)):
        logits[i : i + 1] = dist_f(x[i : i + 1], x_t) / temperature
    logits -= logits.max(1, keepdim=True)[0].detach()
    log_prob = logits - torch.log(logits.exp().sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
    loss = -mean_log_prob_pos.mean()
    stats = {
        "logits/min": logits.min().item(),
        "logits/mean": logits.mean().item(),
    }
    return loss, stats


if __name__ == "__main__":
    cfg: Config = Config().parse_args()
    if cfg.local_rank == 0:
        wandb.init(project="hyp_metric", config=cfg.as_dict())

    torch.cuda.set_device(cfg.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    world_size = torch.distributed.get_world_size()

    train_tr = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=PIL.Image.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    ds_f = {"CUB": CUBirds, "SOP": SOP}
    ds_train = ds_f[cfg.ds](cfg.path, "train", train_tr)
    sampler = torch.utils.data.distributed.DistributedSampler(ds_train)
    dl_train = DataLoader(
        dataset=ds_train,
        sampler=sampler,
        batch_size=cfg.bs,
        num_workers=multiprocessing.cpu_count() // world_size,
        pin_memory=True,
        drop_last=True,
    )
    if cfg.local_rank == 0:
        print(f"iter per epoch: {len(dl_train)}")

    if cfg.model.startswith("dino"):
        body = torch.hub.load("facebookresearch/dino:main", cfg.model)
    else:
        body = timm.create_model(cfg.model, pretrained=True)

    if cfg.hyp_c > 0:
        last = hypnn.ToPoincare(c=cfg.hyp_c, ball_dim=cfg.emb, riemannian=False)
    else:
        last = NormLayer()
    head = nn.Sequential(nn.Linear(384, cfg.emb), last)
    body.head = nn.Identity()
    if cfg.freeze is not None:
        freeze_model(body, cfg.freeze)
    model = HeadSwitch(body, head)
    model.cuda().train()

    model_t = copy.deepcopy(model)
    model_t.cuda().train()
    for param in model_t.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=16.0)
    model_t = amp.initialize(model_t, opt_level="O2", loss_scale=16.0)
    model = DistributedDataParallel(model, delay_allreduce=True)
    loss_f = partial(get_loss, temperature=cfg.t, hyp_c=cfg.hyp_c)

    def update_target(tau):
        for t, s in zip(model_t.parameters(), model.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))

    update_target(0)
    z_queue = deque(maxlen=cfg.qlen)
    y_queue = deque(maxlen=cfg.qlen)

    cudnn.benchmark = True
    for ep in trange(cfg.ep):
        sampler.set_epoch(ep)
        stats_ep = []
        for x, y in dl_train:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            if len(z_queue):
                z = model(x)
                zz, yy = torch.cat(tuple(z_queue)), torch.cat(tuple(y_queue))
                loss, stats = loss_f(z, zz, y, yy)

                optimizer.zero_grad()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 3)
                optimizer.step()
                stats_ep.append({**stats, "loss": loss.item()})
                update_target(cfg.tau)

            with torch.no_grad():
                z = model_t(x)
            all_z = [torch.zeros_like(z) for _ in range(world_size)]
            all_y = [torch.zeros_like(y) for _ in range(world_size)]
            torch.distributed.all_gather(all_z, z)
            torch.distributed.all_gather(all_y, y)
            z = torch.cat(all_z)
            y = torch.cat(all_y)
            z_queue.append(z)
            y_queue.append(y)

        if (ep + 1) % cfg.eval_ep == 0:
            eval_head = get_emb(model, cfg.ds, cfg.path, "eval", world_size)
            eval_body = get_emb(model, cfg.ds, cfg.path, "eval", world_size, True)
            eval_head_t = get_emb(model_t, cfg.ds, cfg.path, "eval", world_size)
            eval_body_t = get_emb(model_t, cfg.ds, cfg.path, "eval", world_size, True)

        if cfg.local_rank == 0:
            stats_ep = {k: np.mean([x[k] for x in stats_ep]) for k in stats_ep[0]}
            if (ep + 1) % cfg.eval_ep == 0:
                stats_ep = {
                    "recall": get_recall(*eval_head, cfg.ds, cfg.hyp_c),
                    "recall_b": get_recall(*eval_body, cfg.ds, 0),
                    "recall_t": get_recall(*eval_head_t, cfg.ds, cfg.hyp_c),
                    "recall_bt": get_recall(*eval_body_t, cfg.ds, 0),
                    **stats_ep
                }
            wandb.log({**stats_ep, "ep": ep})
