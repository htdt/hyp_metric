import torch
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from apex import amp
from apex.parallel import DistributedDataParallel

import os
from tqdm import trange
import wandb
import multiprocessing
from functools import partial
import numpy as np
import PIL
from tap import Tap
from typing_extensions import Literal

from sampler import UniqueClassSempler
from helpers import get_emb, evaluate
from proxy_anchor.dataset import CUBirds, SOP, Cars
from proxy_anchor.dataset.Inshop import Inshop_Dataset
from hyptorch.pmath import dist_matrix
from model import init_model


# flake8: noqa: E501
class Config(Tap):
    path: str = "/home/i"  # path to datasets
    ds: Literal["SOP", "CUB", "Cars", "Inshop"] = "SOP"  # dataset name
    num_samples: int = 2  # how many samples per each category in batch
    bs: int = 900  # batch size per GPU, e.g. --num_samples 3 --bs 900 means each iteration we sample 300 categories with 3 samples
    lr: float = 1e-5  # learning rate
    t: float = 0.2  # cross-entropy temperature
    emb: int = 128  # output embedding size
    freeze: int = 0  # number of blocks in transformer to freeze, None - freeze nothing, 0 - freeze only patch_embed
    ep: int = 100  # number of epochs
    hyp_c: float = 0.1  # hyperbolic c, "0" enables sphere mode
    eval_ep: str = "[100]"  # epochs for evaluation, [] or range "r(start,end,step)", e.g. "r(10,70,20)+[200]" means 10, 30, 50, 200"""
    model: str = "dino_vits16"  # model name from timm or torch.hub, i.e. deit_small_distilled_patch16_224, vit_small_patch16_224, dino_vits16
    save_emb: bool = False  # save embeddings of the dataset after training
    emb_name: str = "emb"  # filename for embeddings
    clip_r: float = 2.3  # feature clipping radius
    resize: int = 224  # image resize
    crop: int = 224  # center crop after resize
    local_rank: int = 0  # set automatically for distributed training


def contrastive_loss(x0, x1, tau, hyp_c):
    # x0 and x1 - positive pair
    # tau - temperature
    # hyp_c - hyperbolic curvature, "0" enables sphere mode

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

    world_size = int(os.environ.get("WORLD_SIZE", 1))
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
            T.RandomResizedCrop(
                cfg.crop, scale=(0.2, 1.0), interpolation=PIL.Image.BICUBIC
            ),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(*mean_std),
        ]
    )

    ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset}
    ds_class = ds_list[cfg.ds]
    ds_train = ds_class(cfg.path, "train", train_tr)
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

    model = init_model(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    if world_size > 1:
        model = DistributedDataParallel(model, delay_allreduce=True)

    loss_f = partial(contrastive_loss, tau=cfg.t, hyp_c=cfg.hyp_c)
    get_emb_f = partial(
        get_emb,
        model=model,
        ds=ds_class,
        path=cfg.path,
        mean_std=mean_std,
        world_size=world_size,
        resize=cfg.resize,
        crop=cfg.crop,
    )
    eval_ep = eval(cfg.eval_ep.replace("r", "list(range").replace(")", "))"))

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
            if world_size > 1:
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

        if (ep + 1) in eval_ep:
            rh, rb = evaluate(get_emb_f, cfg.ds, cfg.hyp_c)

        if cfg.local_rank == 0:
            stats_ep = {k: np.mean([x[k] for x in stats_ep]) for k in stats_ep[0]}
            if (ep + 1) in eval_ep:
                stats_ep = {"recall": rh, "recall_b": rb, **stats_ep}
            wandb.log({**stats_ep, "ep": ep})

    if cfg.save_emb:
        ds_type = "gallery" if cfg.ds == "Inshop" else "eval"
        x, y = get_emb_f(ds_type=ds_type)
        x, y = x.float().cpu(), y.long().cpu()
        torch.save((x, y), cfg.path + "/" + cfg.emb_name + "_eval.pt")

        x, y = get_emb_f(ds_type="train")
        x, y = x.float().cpu(), y.long().cpu()
        torch.save((x, y), cfg.path + "/" + cfg.emb_name + "_train.pt")
