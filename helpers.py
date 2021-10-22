import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from proxy_anchor.utils import calc_recall_at_k
from hyptorch.pmath import dist_matrix
import PIL
import multiprocessing


class MultiSample:
    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))


def get_recall(x, y, ds_name, hyp_c):
    if ds_name == "CUB":
        k_list = [1, 2, 4, 8, 16, 32]
    elif ds_name == "SOP":
        k_list = [1, 10, 100, 1000]

    if hyp_c > 0:
        dist_m = torch.empty(len(x), len(x), device="cuda")
        for i in range(len(x)):
            dist_m[i : i + 1] = dist_matrix(x[i : i + 1], x, hyp_c)
    else:
        dist_m = x @ x.T

    y_cur = y[dist_m.topk(1 + max(k_list), largest=(hyp_c == 0))[1][:, 1:]]
    y = y.cpu()
    y_cur = y_cur.float().cpu()
    recall = [calc_recall_at_k(y, y_cur, k) for k in k_list]
    print(recall)
    return recall[0]


def get_emb(model, ds, path, mean_std, ds_type="eval", world_size=1, skip_head=False):
    eval_tr = T.Compose(
        [
            T.Resize(224, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(*mean_std),
        ]
    )
    ds_eval = ds(path, ds_type, eval_tr)
    if world_size == 1:
        sampler = None
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(ds_eval)
    dl_eval = DataLoader(
        dataset=ds_eval,
        batch_size=100,
        shuffle=False,
        num_workers=multiprocessing.cpu_count() // world_size,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
    )
    model.eval()
    x, y = eval_dataset(model, dl_eval, skip_head)
    y = y.cuda()
    if world_size > 1:
        all_x = [torch.zeros_like(x) for _ in range(world_size)]
        all_y = [torch.zeros_like(y) for _ in range(world_size)]
        torch.distributed.all_gather(all_x, x)
        torch.distributed.all_gather(all_y, y)
        x, y = torch.cat(all_x), torch.cat(all_y)
    model.train()
    return x, y


def eval_dataset(model, dl, skip_head):
    all_x, all_y = [], []
    for x, y in dl:
        with torch.no_grad():
            x = x.cuda(non_blocking=True)
            all_x.append(model(x, skip_head=skip_head))
        all_y.append(y)
    return torch.cat(all_x), torch.cat(all_y)
