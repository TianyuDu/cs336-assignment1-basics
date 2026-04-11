import torch
from torch import nn, Tensor
from einops import einsum
from typing import Optional
from jaxtyping import Float

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Compute root mean square along the last dimension using einsum
        sq = x ** 2  # [..., d]
        mean_sq = sq.mean(dim=-1, keepdim=True)  # [..., 1]
        rms = (mean_sq + self.eps).sqrt()  # [..., 1]

        x_norm = x / rms  # [..., d]
        # Apply weight via einsum
        x_norm = einsum(x_norm, self.weight, "... d, d -> ... d")  # [..., d]
        return x_norm.to(in_dtype)