import torch
from tap import Tap
from typing_extensions import Literal
from helpers import get_emb
from proxy_anchor.dataset import CUBirds, SOP, Cars
from proxy_anchor.dataset.Inshop import Inshop_Dataset
from model import init_model


class Config(Tap):
    path: str = "/home/i"  # path to dataset
    ds: Literal["SOP", "CUB", "Cars", "Inshop"] = "SOP"  # dataset name
    model: str = "dino_vits16"  # model name (see train.py)


def delta_hyp(dismat):
    p = 0
    row = dismat[p, :][None, :]
    col = dismat[:, p][:, None]
    XY_p = 0.5 * (row + col - dismat)
    maxmin = torch.minimum(XY_p[:, :, None], XY_p[None, :, :]).max(1).values
    return (maxmin - XY_p).max()


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
    assert len(emb) > 2000

    result = []
    for i in range(100):
        idx = torch.randperm(len(emb))[:2000]
        emb_cur = emb[idx]
        dists = torch.cdist(emb_cur, emb_cur)
        delta = delta_hyp(dists)
        diam = dists.max()
        rel_delta = (2 * delta) / diam
        result.append(rel_delta)
    rel_delta_mean = torch.tensor(result).mean().item()
    c = (0.144 / rel_delta_mean) ** 2
    print(f"δ = {rel_delta_mean:.3f}, c = {c:.3f}")
