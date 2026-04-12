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
    tokens = torch.as_tensor(x, dtype=torch.long)

    num_valid_starts = len(tokens) - context_length
    context_starts = torch.randint(num_valid_starts, (batch_size,))
    context_offsets = torch.arange(context_length)

    input_positions = context_starts[:, None] + context_offsets
    target_positions = input_positions + 1

    inputs = tokens[input_positions]
    targets = tokens[target_positions]
    return inputs.to(device), targets.to(device)