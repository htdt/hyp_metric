from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from proxy_anchor.dataset import SOP, CUBirds
from proxy_anchor.utils import evaluate_cos, evaluate_cos_SOP
from proxy_anchor.utils import predict_batchwise, calc_recall_at_k
from hyptorch.pmath import dist_matrix


class MultiSample:
    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))


def ss_loss(x, x_t, y, y_t, temperature=0.07):
    """
    x - batch of embeddings,
    x_t - target embeddings,
    y - labels for x,
    y_t - labels for x_t,
    temperature - logits coef,
    based on https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    """

    mask = torch.eq(y.view(-1, 1), y_t.view(1, -1)).float().cuda()

    anchor_dot_contrast = x @ x_t.T / temperature
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
    loss = -mean_log_prob_pos.mean()
    stats = {
        "logits/min": logits.min().item(),
        "logits/mean": logits.mean().item(),
    }
    return loss, stats


def evaluate_hyp(model, dataloader, hyp_c, k_list=[1]):
    X, T = predict_batchwise(model, dataloader)
    dist_m = torch.empty(len(X), len(X), device="cuda")
    for i in range(len(X)):
        dist_m[i : i + 1] = dist_matrix(X[i : i + 1], X, hyp_c)
    Y = T[dist_m.topk(1 + 32, largest=False)[1][:, 1:]]
    Y = Y.float().cpu()
    recall = [calc_recall_at_k(T, Y, k) for k in k_list]
    print(recall)
    return recall


class PadIfNeeded(torch.nn.Module):
    def forward(self, img):
        w, h = TF._get_image_size(img)
        if w < 224 or h < 224:
            padding = [max(0, 224 - w), max(0, 224 - h)]
            img = TF.pad(img, padding, 0, "edge")
        return img


def eval_model(model, ds_name, hyp_c=None):
    if ds_name == "CUB":
        ds = CUBirds
        eval_f = evaluate_cos
        k_list = [1, 2, 4, 8, 16, 32]
    elif ds_name == "SOP":
        ds = SOP
        eval_f = evaluate_cos_SOP
        k_list = [1, 10, 100, 1000]

    if hyp_c is not None:
        eval_f = partial(evaluate_hyp, hyp_c=hyp_c, k_list=k_list)

    eval_tr = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds_eval = ds("/home/i", "eval", eval_tr)
    dl_eval = DataLoader(
        dataset=ds_eval,
        batch_size=100,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        drop_last=False,
    )
    model.eval()
    r = eval_f(model, dl_eval)
    model.train()
    return r[0]


def freeze_model(model, num_block):
    def fr(m):
        for param in m.parameters():
            param.requires_grad = False

    fr(model.patch_embed)
    fr(model.pos_drop)
    for i in range(num_block):
        fr(model.blocks[i])


class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)
