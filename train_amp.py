import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import apex
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

from sampler import UniqueClassSempler
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
    emb: int = 128
    freeze: int = None
    ep: int = 10
    hyp_c: float = 0.1
    eval_ep: int = 10
    model: str = "dino_vits16"
    save_emb: bool = False
    emb_name: str = "emb"
    local_rank: int = 0


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
            # T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    ds_f = {"CUB": CUBirds, "SOP": SOP}
    ds_train = ds_f[cfg.ds](cfg.path, "train", train_tr)
    assert len(ds_train.ys) * cfg.num_samples >= cfg.bs * world_size
    sampler = UniqueClassSempler(
        ds_train.ys, cfg.num_samples, cfg.local_rank, world_size
    )
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
    model = apex.parallel.convert_syncbn_model(model)
    model.cuda().train()

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=16.0)
    model = DistributedDataParallel(model, delay_allreduce=True)

    loss_f = partial(contrastive_loss, tau=cfg.t, hyp_c=cfg.hyp_c)

    cudnn.benchmark = True
    for ep in trange(cfg.ep):
        sampler.set_epoch(ep)
        stats_ep = []
        for x, y in dl_train:
            y = y.view(len(y) // cfg.num_samples, cfg.num_samples)
            assert (y[:, 0] == y[:, -1]).all()
            s = y[:, 0].tolist()
            assert len(set(s)) == len(s)

            x = x.cuda(non_blocking=True)
            z = model(x).view(len(x) // cfg.num_samples, cfg.num_samples, cfg.emb)
            with torch.no_grad():
                all_z = [torch.zeros_like(z) for _ in range(world_size)]
                torch.distributed.all_gather(all_z, z)
            all_z[cfg.local_rank] = z
            z = torch.cat(all_z)
            loss = 0
            for i in range(cfg.num_samples):
                for j in range(cfg.num_samples):
                    if i != j:
                        l, s = loss_f(z[:, i], z[:, j])
                        loss += l
                        stats_ep.append({**s, "loss": l.item()})

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 3)
            optimizer.step()

        if (ep + 1) % cfg.eval_ep == 0:
            eval_head = get_emb(model, cfg.ds, cfg.path, "eval", world_size)
            eval_body = get_emb(model, cfg.ds, cfg.path, "eval", world_size, True)

        if cfg.local_rank == 0:
            stats_ep = {k: np.mean([x[k] for x in stats_ep]) for k in stats_ep[0]}
            if (ep + 1) % cfg.eval_ep == 0:
                recall_head = get_recall(*eval_head, cfg.ds, cfg.hyp_c)
                recall_body = get_recall(*eval_body, cfg.ds, 0)
                stats_ep = {"recall": recall_head, "recall_b": recall_body, **stats_ep}
            wandb.log({**stats_ep, "ep": ep})

    if cfg.save_emb:
        x, y = get_emb(model, cfg.ds, cfg.path, "eval", world_size)
        x, y = x.float().cpu(), y.long().cpu()
        torch.save((x, y), cfg.path + "/" + cfg.emb_name + "_eval.pt")

        x, y = get_emb(model, cfg.ds, cfg.path, "train", world_size)
        x, y = x.float().cpu(), y.long().cpu()
        torch.save((x, y), cfg.path + "/" + cfg.emb_name + "_train.pt")
