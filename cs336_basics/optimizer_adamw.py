import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        beta1, beta2 = betas
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                t = state["t"] + 1
                m = state["m"]
                v = state["v"]
                step_size = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                with torch.no_grad():
                    p.add_(p, alpha=-lr * weight_decay)
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    p.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)

                state["t"] = t

        return loss