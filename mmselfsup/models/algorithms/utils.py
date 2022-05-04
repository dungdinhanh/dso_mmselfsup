import torch.nn as nn


def cosine_sim(input, target, mean=True):
    target = target.detach()
    pred_norm = nn.functional.normalize(input, dim=1)
    target_norm = nn.functional.normalize(target, dim=1)
    cs_sim = -(pred_norm * target_norm).sum(dim=1)
    if mean:
        loss = cs_sim.mean()
        return loss
    else:
        return cs_sim