import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Int
from torch import Tensor


def data_loading(
    x: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    num_valid_starts = len(x) - context_length
    context_starts = np.random.randint(num_valid_starts, size=batch_size)
    context_offsets = np.arange(context_length)

    input_positions = context_starts[:, None] + context_offsets
    target_positions = input_positions + 1

    inputs = torch.as_tensor(x[input_positions], dtype=torch.long, device=device)
    targets = torch.as_tensor(x[target_positions], dtype=torch.long, device=device)
    return inputs, targets