from functools import partial
from tap import Tap
from typing_extensions import Literal
from helpers import get_emb, get_recall, get_recall_inshop
from proxy_anchor.dataset import CUBirds, SOP, Cars
from proxy_anchor.dataset.Inshop import Inshop_Dataset
from model import init_model


class Config(Tap):
    path: str = "/home/i"  # path to dataset
    ds: Literal["SOP", "CUB", "Cars", "Inshop"] = "SOP"  # dataset name
    model: str = "dino_vits16"  # model name (see train.py)
    resize: int = 224  # image resize
    crop: int = 224  # center crop after resize


if __name__ == "__main__":
    cfg: Config = Config().parse_args()
    ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset}
    if cfg.model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    model = init_model(cfg)
    get_emb_f = partial(
        get_emb,
        model=model,
        ds=ds_list[cfg.ds],
        path=cfg.path,
        mean_std=mean_std,
        world_size=1,
        resize=cfg.resize,
        crop=cfg.crop,
        skip_head=True,
    )
    if cfg.ds != "Inshop":
        emb = get_emb_f(ds_type="eval")
        get_recall(*emb, cfg.ds, 0)
    else:
        emb_query = get_emb_f(ds_type="query")
        emb_gal = get_emb_f(ds_type="gallery")
        get_recall_inshop(*emb_query, *emb_gal, 0)
