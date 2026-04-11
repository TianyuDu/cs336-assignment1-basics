import torch
from torch import nn, Tensor
from jaxtyping import Float
from typing import Optional
from einops import einsum

class Linear(nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(
            out_features,
            in_features,
            device=device,
            dtype=dtype,
        ))
        # Initialize weights with truncated normal distribution
        # Mean: 0,  Std: sqrt(2 / (in_features + out_features)), Truncated at [-3σ, 3σ]
        std = (2.0 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: Float[Tensor, " ... in_features"]) -> Float[Tensor, " ... out_features"]:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")