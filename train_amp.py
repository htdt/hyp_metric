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
import pickle
import numpy as np
import os
import PIL

from sampler import UniqueClassSempler
from utils import get_eval_emb, get_recall, freeze_model, NormLayer
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
    if cfg.local_rank == 0:
        wandb.init(project="hyp_metric", config=cfg)

    torch.cuda.set_device(cfg.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    world_size = torch.distributed.get_world_size()

    train_tr = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=PIL.Image.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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
    sampler = UniqueClassSempler(lbl, 2, cfg.local_rank, world_size)
    dl_train = DataLoader(
        dataset=ds_train,
        sampler=sampler,
        batch_size=cfg.bs,
        num_workers=multiprocessing.cpu_count() // world_size,
        pin_memory=True,
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
    model = apex.parallel.convert_syncbn_model(model)
    model.cuda().train()

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    model = DistributedDataParallel(model, delay_allreduce=True)

    loss_f = partial(contrastive_loss, tau=cfg.t, hyp_c=cfg.hyp_c)

    cudnn.benchmark = True
    for ep in trange(cfg.ep):
        sampler.set_epoch(ep)
        stats_ep = []
        for x, y in dl_train:
            y = y.view(len(y) // 2, 2)
            assert (y[:, 0] == y[:, 1]).all()
            s = y[:, 0].tolist()
            assert len(set(s)) == len(s)

            x = x.cuda(non_blocking=True)
            z = model(x).view(len(x) // 2, 2, cfg.emb)
            with torch.no_grad():
                all_z = [torch.zeros_like(z) for _ in range(world_size)]
                torch.distributed.all_gather(all_z, z)
            all_z[cfg.local_rank] = z
            z = torch.cat(all_z)
            loss0, stats0 = loss_f(z[:, 0], z[:, 1])
            loss1, stats1 = loss_f(z[:, 1], z[:, 0])

            optimizer.zero_grad()
            with amp.scale_loss(loss0 + loss1, optimizer) as scaled_loss:
                scaled_loss.backward()
            # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 3)
            optimizer.step()
            stats_ep.append({**stats0, "loss": loss0.item()})
            stats_ep.append({**stats1, "loss": loss1.item()})

        if (ep + 1) % cfg.eval_ep == 0:
            eval_data = get_eval_emb(model, cfg.ds, cfg.path, world_size)

        if cfg.local_rank == 0:
            stats_ep = {k: np.mean([x[k] for x in stats_ep]) for k in stats_ep[0]}
            if (ep + 1) % cfg.eval_ep == 0:
                recall = get_recall(*eval_data, cfg.ds, cfg.hyp_c)
                stats_ep = {"recall": recall, **stats_ep}
            wandb.log({**stats_ep, "ep": ep})

    # model.head = NormLayer()
    # eval_data = get_eval_emb(model, cfg.ds, cfg.path, world_size)
    # print(eval_data[0].shape, eval_data[1].shape)
    # if cfg.local_rank == 0:
    #     recall = get_recall(*eval_data, cfg.ds, 0)
    #     wandb.log({"recall_b": recall})
