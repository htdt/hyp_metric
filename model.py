import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import hyptorch.nn as hypnn


def init_model(cfg):
    if cfg.model.startswith("dino"):
        body = torch.hub.load("facebookresearch/dino:main", cfg.model)
    else:
        body = timm.create_model(cfg.model, pretrained=True)
    if cfg.hyp_c > 0:
        last = hypnn.ToPoincare(c=cfg.hyp_c, ball_dim=cfg.emb, riemannian=False)
    else:
        last = NormLayer()
    bdim = 2048 if cfg.model == "resnet50" else 384
    head = nn.Sequential(nn.Linear(bdim, cfg.emb, bias=False), last)
    nn.init.orthogonal_(head[0].weight.data)
    rm_head(body)
    if cfg.freeze is not None:
        freeze(body, cfg.freeze)
    model = HeadSwitch(body, head)
    model.cuda().train()
    return model


class HeadSwitch(nn.Module):
    def __init__(self, body, head):
        super(HeadSwitch, self).__init__()
        self.body = body
        self.head = head
        self.norm = NormLayer()

    def forward(self, x, skip_head=False):
        x = self.body(x)
        if type(x) == tuple:
            x = x[0]
        if not skip_head:
            x = self.head(x)
        else:
            x = self.norm(x)
        return x


class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


def freeze(model, num_block):
    def fr(m):
        for param in m.parameters():
            param.requires_grad = False

    fr(model.patch_embed)
    fr(model.pos_drop)
    for i in range(num_block):
        fr(model.blocks[i])


def rm_head(m):
    names = set(x[0] for x in m.named_children())
    target = {"head", "fc", "head_dist"}
    for x in names & target:
        m.add_module(x, nn.Identity())
