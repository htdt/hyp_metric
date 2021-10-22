import torch
from tap import Tap
from typing_extensions import Literal
from helpers import get_emb, get_recall
from proxy_anchor.dataset import CUBirds, SOP
from model import init_model


class Config(Tap):
    path: str = "/home/i"
    ds: Literal["SOP", "CUB"] = "SOP"
    emb: int = 128
    hyp_c: float = 0.1
    model: str = "dino_vits16"


if __name__ == "__main__":
    cfg: Config = Config().parse_args()
    torch.cuda.set_device(0)

    ds_f = {"CUB": CUBirds, "SOP": SOP}
    ds = ds_f[cfg.ds]
    if cfg.model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    model = init_model(cfg)
    eval_head = get_emb(model, ds, cfg.path, mean_std, "eval", 1)
    eval_body = get_emb(model, ds, cfg.path, mean_std, "eval", 1, True)
    get_recall(*eval_head, cfg.ds, cfg.hyp_c)
    get_recall(*eval_body, cfg.ds, 0)
