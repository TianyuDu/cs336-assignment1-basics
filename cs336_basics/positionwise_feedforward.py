import torch
from torch import nn, Tensor
from einops import einsum
from typing import Optional
from jaxtyping import Float

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
    ):
        super().__init__()
        # If d_ff is not provided, use the assignment default hidden size.
        if d_ff is None:
            d_ff = int(round((8 / 3) * d_model / 64) * 64)

        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model))

        # initialize weights with torch.nn.init.trunc_normal_ per specified std
        std1 = (2 / (self.W1.shape[0] + self.W1.shape[1])) ** 0.5
        std2 = (2 / (self.W2.shape[0] + self.W2.shape[1])) ** 0.5
        std3 = (2 / (self.W3.shape[0] + self.W3.shape[1])) ** 0.5

        nn.init.trunc_normal_(self.W1, mean=0.0, std=std1, a=-3*std1, b=3*std1)
        nn.init.trunc_normal_(self.W2, mean=0.0, std=std2, a=-3*std2, b=3*std2)
        nn.init.trunc_normal_(self.W3, mean=0.0, std=std3, a=-3*std3, b=3*std3)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        # SLU(W1@x)
        slu_left = einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")  # W1@x
        slu_left = slu_left * torch.sigmoid(slu_left)  # SLU(W1@x)
        slu_right = einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")  # W3@x
        slu_inner = slu_left * slu_right  # (... d_ff)  SLU(W1@x) * W3@x
        return einsum(slu_inner, self.W2, "... d_ff, d_model d_ff -> ... d_model")  # W2@slu_inner

