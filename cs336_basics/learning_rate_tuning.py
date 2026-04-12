import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch
from torch import Tensor


class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Tensor | None:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                step_size = lr / math.sqrt(t + 1)

                with torch.no_grad():
                    p -= step_size * p.grad

                state["t"] = t + 1

        return loss


def run_training(lr: float, num_steps: int = 10, seed: int = 0) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10), generator=generator))
    opt = SGD([weights], lr=lr)

    losses: list[float] = []
    for _ in range(num_steps):
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.item())
        loss.backward()
        opt.step()

    return losses


def run_learning_rate_experiment(
    learning_rates: Iterable[float] = (1.0, 1e1, 1e2, 1e3),
    num_steps: int = 10,
    seed: int = 0,
) -> dict[float, list[float]]:
    return {
        lr: run_training(lr=lr, num_steps=num_steps, seed=seed)
        for lr in learning_rates
    }


def main() -> None:
    results = run_learning_rate_experiment()
    for lr, losses in results.items():
        lr_label = "1" if lr == 1.0 else f"{lr:.0e}"
        formatted_losses = ", ".join(f"{loss:.6f}" for loss in losses)
        print(f"lr={lr_label}: [{formatted_losses}]")


if __name__ == "__main__":
    main()