from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.softmax import softmax


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

        std = (2.0 / (d_model + vocab_size)) ** 0.5
        nn.init.trunc_normal_(self.lm_head.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(
        self,
        token_ids: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length vocab_size"]:
        # token_ids: [..., sequence_length]
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds context length {self.context_length}."
            )

        # Turn token ids into vectors of size d_model.
        # x: [..., sequence_length, d_model]
        x = self.token_embeddings(token_ids)

        # Build positions 0, 1, ..., sequence_length - 1 once and reuse them in every block.
        # leading_shape is the batch shape, so x still has shape [..., sequence_length, d_model].
        *leading_shape, sequence_length, _ = x.shape
        view_shape = (1,) * len(leading_shape) + (sequence_length,)
        # token_positions: [..., sequence_length]
        token_positions = torch.arange(sequence_length, device=x.device).reshape(view_shape)
        token_positions = token_positions.expand(*leading_shape, sequence_length)

        # Run the transformer blocks in order. Each block already does
        # attention + residual and then FFN + residual.
        # x stays [..., sequence_length, d_model] through the whole stack.
        for block in self.transformer_blocks:
            x = block(x, token_positions=token_positions, use_rope=True)

        # Apply the final norm, then map each hidden vector to vocabulary-sized logits.
        # x after ln_final: [..., sequence_length, d_model]
        x = self.ln_final(x)
        # logits: [..., sequence_length, vocab_size]
        logits = self.lm_head(x)
        return logits