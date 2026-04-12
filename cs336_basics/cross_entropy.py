import torch
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, ""]:
    """Compute mean cross-entropy over arbitrary leading dimensions."""
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    log_probs = torch.log_softmax(logits_flat, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)
    return nll.mean()


def perplexity(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, ""]:
    """Compute perplexity as exp(mean cross-entropy)."""
    return torch.exp(cross_entropy(logits, targets))