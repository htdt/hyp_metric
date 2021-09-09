import torch
from hyptorch.pmath import dist_matrix


def hyp_loss(x, x_t, y, y_t, hyp_c, temperature):
    """
    x - batch of embeddings,
    x_t - target embeddings,
    y - labels for x,
    y_t - labels for x_t,
    hyp_c - hyperbolic curvature,
    temperature - logits coef
    """

    mask = torch.eq(y.view(-1, 1), y_t.view(1, -1)).float().cuda()

    dm = torch.empty(len(x), len(x_t), device="cuda")
    for i in range(len(x)):
        dm[i : i + 1] = dist_matrix(x[i : i + 1], x_t, hyp_c)
    dm = -dm / temperature
    # dm = -dist_matrix(x, x_t, hyp_c) / temperature
    logits_max, _ = torch.max(dm, dim=1, keepdim=True)
    logits = dm - logits_max.detach()

    log_prob = logits - torch.log(logits.exp().sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
    loss = -mean_log_prob_pos.mean()
    stats = {
        "logits/min": logits.min().item(),
        "logits/mean": logits.mean().item(),
    }
    return loss, stats
