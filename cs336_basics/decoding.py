from __future__ import annotations

from collections.abc import Sequence

import torch
from jaxtyping import Int
from torch import Tensor, nn

from cs336_basics.softmax import softmax


def decode(
    model: nn.Module,
    prompt_token_ids: Sequence[int] | Int[Tensor, " sequence_length"],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[int]:
    """Generate tokens autoregressively from a prompt."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1].")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    generated = torch.as_tensor(prompt_token_ids, dtype=torch.long, device=device).clone()
    if generated.ndim != 1:
        raise ValueError("prompt_token_ids must be a 1D tensor or a sequence of ints.")
    if generated.numel() == 0:
        raise ValueError("prompt_token_ids must contain at least one token.")

    context_length = getattr(model, "context_length", None)
    if context_length is not None:
        context_length = int(context_length)

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # If the prompt grows past the model's context window, keep
                # only the most recent tokens before the next forward pass.
                if context_length is None:
                    model_input = generated
                else:
                    model_input = generated[-context_length:]

                # logits: [1, current_sequence_length, vocab_size]
                logits = model(model_input.unsqueeze(0))

                # The next-token distribution comes from the final position.
                next_token_logits = logits[0, -1, :]

                if temperature == 0.0:
                    next_token = torch.argmax(next_token_logits, dim=-1)
                else:
                    probabilities = softmax(next_token_logits / temperature, i=-1)

                    if top_p < 1.0:
                        sorted_probabilities, sorted_indices = torch.sort(
                            probabilities,
                            descending=True,
                        )
                        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

                        # Keep the smallest prefix whose cumulative mass reaches top_p.
                        keep_mask = cumulative_probabilities - sorted_probabilities < top_p
                        sorted_probabilities = sorted_probabilities * keep_mask

                        probabilities = torch.zeros_like(probabilities)
                        probabilities[sorted_indices] = sorted_probabilities
                        probabilities = probabilities / torch.sum(probabilities)

                    next_token = torch.multinomial(probabilities, num_samples=1)[0]

                generated = torch.cat((generated, next_token.unsqueeze(0)))
                if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                    break
    finally:
        if was_training:
            model.train()

    return generated.tolist()
