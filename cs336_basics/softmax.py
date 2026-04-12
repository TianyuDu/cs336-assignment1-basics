import torch
from torch import Tensor
from jaxtyping import Float


def softmax(x: Float[Tensor, "... dim"], i: int) -> Float[Tensor, "... dim"]:
    # Subtract the maximum along dimension i for numerical stability.
    shifted = x - torch.max(x, dim=i, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    sum_exp_shifted = torch.sum(exp_shifted, dim=i, keepdim=True)
    return exp_shifted / sum_exp_shifted
