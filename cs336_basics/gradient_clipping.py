from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total_norm_sq = torch.zeros((), device=grads[0].device)
    for grad in grads:
        total_norm_sq += grad.pow(2).sum()

    total_norm = torch.sqrt(total_norm_sq)
    if total_norm <= max_l2_norm:
        return

    scale = max_l2_norm / (total_norm + 1e-6)
    for grad in grads:
        grad.mul_(scale)