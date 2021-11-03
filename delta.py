import torch
from tap import Tap
from typing_extensions import Literal
from helpers import get_emb
from proxy_anchor.dataset import CUBirds, SOP, Cars
from proxy_anchor.dataset.Inshop import Inshop_Dataset
from model import init_model


class Config(Tap):
    path: str = "/home/i"
    ds: Literal["SOP", "CUB", "Cars", "Inshop"] = "SOP"
    emb: int = 128
    hyp_c: float = 0.1
    model: str = "dino_vits16"
    clip_r: int = None
    freeze: int = None


def delta_hyp(dismat):
    p = 0
    row = dismat[p, :][None, :]
    col = dismat[:, p][:, None]
    XY_p = 0.5 * (row + col - dismat)
    maxmin = torch.minimum(XY_p[:, :, None], XY_p[None, :, :]).max(1).values
    return (maxmin - XY_p).max()


def calculate_c(delta, diam):
    rel_delta = (2 * delta) / diam
    return (0.144 / rel_delta) ** 2


if __name__ == "__main__":
    cfg: Config = Config().parse_args()
    ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset}
    ds = ds_list[cfg.ds]
    if cfg.model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    model = init_model(cfg)
    model.head = torch.nn.Identity()
    emb = get_emb(model, ds, cfg.path, mean_std, "train", 1)[0]
    print(len(emb))

    c_list = []
    for i in range(100):
        idx = torch.randperm(len(emb))[:2000]
        emb_cur = emb[idx]
        dists = torch.cdist(emb_cur, emb_cur)
        delta = delta_hyp(dists).item()
        diam = dists.max().item()
        c_list.append(calculate_c(delta, diam))
    print(sum(c_list) / len(c_list))
