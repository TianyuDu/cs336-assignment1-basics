from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


def multihead_self_attention(
    Q: Float[Tensor, " ... num_heads queries d_k"],
    K: Float[Tensor, " ... num_heads keys d_k"],
    V: Float[Tensor, " ... num_heads keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... num_heads queries d_v"]:
    return scaled_dot_product_attention(Q, K, V, mask)


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: Optional[int] = None,
        theta: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        if (max_seq_len is None) != (theta is None):
            raise ValueError("max_seq_len and theta must both be provided to enable RoPE.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.output_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)

        std = (2.0 / (d_model + d_model)) ** 0.5
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.output_proj):
            nn.init.trunc_normal_(proj.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

        self.rope = None
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len,
                device=device,
            )

        # Cache tensors that can be reused across forward passes instead of rebuilding them
        # every time. If we see a longer sequence later, we will grow these caches on demand.
        self.register_buffer("causal_mask", torch.empty(0, 0, dtype=torch.bool, device=device), persistent=False)
        self.register_buffer("default_positions", torch.empty(0, dtype=torch.long, device=device), persistent=False)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
        mask: Bool[Tensor, " ... sequence_length sequence_length"] | None = None,
        use_rope: bool = True,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected input dimension {self.d_model}, got {x.shape[-1]}.")

        # We keep the whole attention pipeline in one function so students can step through
        # it top-to-bottom: project -> split into heads -> optional RoPE -> mask -> attend -> merge.
        *leading_shape, sequence_length, _ = x.shape

        # 1) Do the Q/K/V projections in one large matrix multiply, then split the result
        #    back into three tensors and reshape each one into attention heads.
        qkv_weight = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        qkv = F.linear(x, qkv_weight)
        Q, K, V = qkv.split(self.d_model, dim=-1)

        Q = Q.reshape(*leading_shape, sequence_length, self.num_heads, self.d_head).transpose(-3, -2)
        K = K.reshape(*leading_shape, sequence_length, self.num_heads, self.d_head).transpose(-3, -2)
        V = V.reshape(*leading_shape, sequence_length, self.num_heads, self.d_head).transpose(-3, -2)

        # 2) RoPE is applied to queries and keys only. If positions are not supplied,
        #    use the standard 0, 1, ..., sequence_length - 1 positions.
        if self.rope is not None and use_rope:
            if token_positions is None:
                if self.default_positions.shape[0] < sequence_length or self.default_positions.device != x.device:
                    self.default_positions = torch.arange(sequence_length, device=x.device, dtype=torch.long)

                view_shape = (1,) * len(leading_shape) + (sequence_length,)
                token_positions = self.default_positions[:sequence_length].reshape(view_shape).expand(*leading_shape, sequence_length)

            # Add a singleton head axis so token positions broadcast across all heads.
            rope_positions = token_positions.unsqueeze(-2)
            Q = self.rope(Q, rope_positions)
            K = self.rope(K, rope_positions)

        # 3) Build the causal mask so position i can only attend to positions <= i.
        #    We cache the largest one we have seen so far and slice out the piece we need.
        if self.causal_mask.shape[-1] < sequence_length or self.causal_mask.device != x.device:
            self.causal_mask = torch.tril(
                torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device)
            )

        causal_mask = self.causal_mask[:sequence_length, :sequence_length]
        causal_mask = causal_mask.reshape(*((1,) * (len(leading_shape) + 1)), sequence_length, sequence_length)

        # If the caller also provides a mask, combine it with the causal mask.
        if mask is not None and mask.ndim == x.ndim:
            mask = mask.unsqueeze(-3)

        attn_mask = causal_mask if mask is None else causal_mask & mask

        # 4) Apply attention independently in each head.
        attn_output = multihead_self_attention(Q, K, V, attn_mask)

        # 5) Move the head dimension back next to the feature dimension, flatten the heads,
        #    and apply the final output projection.
        attn_output = attn_output.transpose(-3, -2).contiguous()
        attn_output = attn_output.reshape(*leading_shape, sequence_length, self.d_model)
        return self.output_proj(attn_output)

