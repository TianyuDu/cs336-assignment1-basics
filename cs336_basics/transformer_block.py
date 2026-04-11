from typing import Optional

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from cs336_basics.multihead_self_attentinon import CausalMultiHeadSelfAttention
from cs336_basics.positionwise_feedforward import SwiGLU
from cs336_basics.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: Optional[int] = None,
        theta: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff=d_ff)
        if device is not None or dtype is not None:
            self.ffn = self.ffn.to(device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
        mask: Bool[Tensor, " ... sequence_length sequence_length"] | None = None,
        use_rope: bool = True,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        # Pre-norm attention block: normalize first, run attention, then add the residual.
        attn_input = self.ln1(x)
        attn_output = self.attn(
            attn_input,
            token_positions=token_positions,
            mask=mask,
            use_rope=use_rope,
        )
        x = x + attn_output

        # Pre-norm feed-forward block: normalize, apply the FFN, then add the residual.
        ffn_input = self.ln2(x)
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output

        return x