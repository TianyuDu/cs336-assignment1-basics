import torch
from torch import nn, Tensor
from typing import Optional
from jaxtyping import Float, Int


class RotaryPositionalEmbedding(nn.Module):
    """Educational, single-pass RoPE helper.

    All computation is done inside `forward` so students can read it in one place and
    run the operation step-by-step. The cos/sin tables are precomputed once in `__init__`.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.d_half = self.d_k // 2

        if self.d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE so pairs can be rotated.")

        # Precompute trig tables once up front so every forward call is fast and students
        # can focus on the RoPE math itself.
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.d_half, device=self.device, dtype=torch.float32) / self.d_half)
        )
        positions = torch.arange(self.max_seq_len, device=self.device, dtype=torch.float32)
        angles = torch.einsum("i,j->ij", positions, inv_freq)  # [max_seq_len, d_half]
        self.register_buffer("cos_cached", angles.cos(), persistent=False)  # [max_seq_len, d_half]
        self.register_buffer("sin_cached", angles.sin(), persistent=False)  # [max_seq_len, d_half]

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        # Keep the original dtype so this function is drop-in with mixed-precision models.
        orig_dtype = x.dtype

        # x: [..., seq_len, d_k], token_positions: [..., seq_len]
        *leading_shape, seq_len, d_k = x.shape
        assert d_k == self.d_k, f"d_k mismatch: {d_k} (input) vs {self.d_k} (module)"
        assert token_positions.shape[-1] == seq_len, "token_positions must match sequence length"

        # A quick guard so students see where index errors would happen.
        if token_positions.max() >= self.max_seq_len:
            raise ValueError(
                f"Token positions must be < max_seq_len ({self.max_seq_len}), got max {int(token_positions.max())}"
            )

        # 1) Fetch precomputed cos/sin for each token position in the batch/sequence.
        cos = self.cos_cached[token_positions.to(device=self.cos_cached.device)]
        sin = self.sin_cached[token_positions.to(device=self.cos_cached.device)]

        # 3) Split the feature dimension into even/odd pairs.
        #    [ ... , seq_len, d_k] -> [ ... , seq_len, d_half] + [ ... , seq_len, d_half]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # 4) Apply the RoPE rotation for each pair:
        #    [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos

        # 5) Interleave the rotated pairs back to [..., seq_len, d_k].
        x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1)
        x_rot = x_rot.reshape(*leading_shape, seq_len, d_k)

        # Convert back to the input dtype so the module behaves like the old implementation.
        return x_rot.to(orig_dtype)
