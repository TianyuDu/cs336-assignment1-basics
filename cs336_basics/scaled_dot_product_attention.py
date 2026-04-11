import math

import torch
from torch import Tensor
from jaxtyping import Bool, Float
from cs336_basics.softmax import softmax
from einops import einsum


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]

    # Attention scores: (..., queries, keys)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is None:
        probs = softmax(scores, -1)
    else:
        # Keep masked positions out of the max and final normalization.
        masked_scores = scores.masked_fill(~mask, -1e9)
        shifted = masked_scores - masked_scores.max(dim=-1, keepdim=True).values
        exp_shifted = torch.exp(shifted) * mask
        probs = exp_shifted / exp_shifted.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # Weighted sum of values: (..., queries, d_v)
    return einsum(probs, V, "... queries keys, ... keys d_v -> ... queries d_v")  # probs @ V