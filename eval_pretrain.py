import torch
import torch.nn as nn
import timm
from tap import Tap
from typing_extensions import Literal
from helpers import get_emb, get_recall, NormLayer, HeadSwitch
import hyptorch.nn as hypnn


class Config(Tap):
    path: str = "/home/i"
    ds: Literal["SOP", "CUB"] = "SOP"
    emb: int = 128
    hyp_c: float = 0.1
    model: str = "dino_vits16"


if __name__ == "__main__":
    cfg: Config = Config().parse_args()
    torch.cuda.set_device(0)

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
    model = HeadSwitch(body, head)
    model.cuda()

    eval_head = get_emb(model, cfg.ds, cfg.path, "eval", 1)
    eval_body = get_emb(model, cfg.ds, cfg.path, "eval", 1, True)
    get_recall(*eval_head, cfg.ds, cfg.hyp_c)
    get_recall(*eval_body, cfg.ds, 0)
