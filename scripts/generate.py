"""Load a TinyStories checkpoint and sample text from it.

Examples:
  uv run python scripts/generate.py
  uv run python scripts/generate.py --prompt "Once upon a time"
  uv run python scripts/generate.py --temperature 0.7 --top-p 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs336_basics.decoding import decode
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training_launcher_utils import SECTION_72_MODEL_CONFIG, detect_default_device
from cs336_basics.transformer_lm import TransformerLM

SPECIAL_TOKEN = "<|endoftext|>"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "723-learning-rate-tuning" / "0_001.pt"
DEFAULT_VOCAB_PATH = REPO_ROOT / "data" / "bpe_tinystories" / "vocab.json"
DEFAULT_MERGES_PATH = REPO_ROOT / "data" / "bpe_tinystories" / "merges.txt"


def generate_text(
    *,
    checkpoint_path: Path,
    vocab_path: Path,
    merges_path: Path,
    special_token: str = SPECIAL_TOKEN,
    prompt: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    device: str = detect_default_device(),
    dtype: str = "float32",
    vocab_size: int = int(SECTION_72_MODEL_CONFIG["vocab_size"]),
    context_length: int = 256,
    d_model: int = int(SECTION_72_MODEL_CONFIG["d_model"]),
    num_layers: int = int(SECTION_72_MODEL_CONFIG["num_layers"]),
    num_heads: int = int(SECTION_72_MODEL_CONFIG["num_heads"]),
    d_ff: int = int(SECTION_72_MODEL_CONFIG["d_ff"]),
    rope_theta: float = float(SECTION_72_MODEL_CONFIG["rope_theta"]),
) -> dict[str, Any]:
    """Run the whole generation pipeline in one place for clarity."""
    checkpoint_path = checkpoint_path.expanduser()
    vocab_path = vocab_path.expanduser()
    merges_path = merges_path.expanduser()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Tokenizer vocab not found: {vocab_path}")
    if not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer merges not found: {merges_path}")
    if max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device but CUDA is not available.")
    if device.startswith("mps") and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("Requested MPS device but MPS is not available.")

    torch_dtype_map: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in torch_dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Fix the RNG so temperature/top-p sampling is reproducible.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load the TinyStories tokenizer and keep <|endoftext|> as one token.
    print(f"Loading tokenizer from {vocab_path} ...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(vocab_path),
        merges_filepath=str(merges_path),
        special_tokens=[special_token],
    )
    eos_token_id = tokenizer.special_token_to_id[special_token]

    # An empty prompt cannot be decoded autoregressively, so start from the
    # end-of-text token to ask the model for a fresh story.
    if prompt:
        prompt_token_ids = tokenizer.encode(prompt)
    else:
        prompt_token_ids = [eos_token_id]
    if not prompt_token_ids:
        raise ValueError("Prompt must encode to at least one token.")

    # Rebuild the same model architecture used during training.
    print(f"Building model (d_model={d_model}, num_layers={num_layers}, num_heads={num_heads}) ...")
    device_obj = torch.device(device)
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device_obj,
        dtype=torch_dtype_map[dtype],
    )

    # Training checkpoints store more than just the model, so peel out the
    # weight dictionary and then load it into the freshly constructed model.
    print(f"Loading checkpoint from {checkpoint_path} ...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint from {checkpoint_path}. "
            "The file may be truncated or corrupted, so try another checkpoint path."
        ) from exc
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        checkpoint_iteration = checkpoint.get("iteration")
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        checkpoint_iteration = None
    else:
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

    # Some checkpoints may come from torch.compile and include an _orig_mod.
    # prefix; stripping it keeps the loader student-friendly and robust.
    cleaned_state_dict = {
        str(key).removeprefix("_orig_mod."): value for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict)
    print("Checkpoint loaded successfully.")

    # Sample a continuation with the assignment decoder implementation.
    print(f"Generating up to {max_new_tokens} tokens (temperature={temperature}, top_p={top_p}) ...")
    full_token_ids = decode(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
    )

    completion_token_ids = full_token_ids[len(prompt_token_ids) :]
    stopped_on_eos = bool(completion_token_ids and completion_token_ids[-1] == eos_token_id)

    # Hide the synthetic start token for empty-prompt sampling, and omit the
    # trailing <|endoftext|> token from the printed text dump.
    display_token_ids = list(full_token_ids)
    if not prompt:
        display_token_ids = display_token_ids[1:]
    if stopped_on_eos and display_token_ids:
        display_token_ids = display_token_ids[:-1]
    generated_text = tokenizer.decode(display_token_ids)

    print(f"checkpoint_path: {checkpoint_path}")
    if checkpoint_iteration is not None:
        print(f"checkpoint_iteration: {checkpoint_iteration}")
    print(f"device: {device_obj}")
    print(f"dtype: {dtype}")
    print(f"temperature: {temperature}")
    print(f"top_p: {top_p}")
    print(f"seed: {seed}")
    if prompt:
        print(f"prompt: {prompt!r}")
        print(f"prompt_tokens: {len(prompt_token_ids)}")
    else:
        print(f"prompt: {special_token!r} (used internally to start a fresh sample)")
        print("prompt_tokens: 0 user tokens")
    print(f"generated_tokens: {len(completion_token_ids)}")
    print(f"stopped_on_eos: {stopped_on_eos}")
    print("\nGenerated text:\n")
    print(generated_text)

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_iteration": checkpoint_iteration,
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_token_ids,
        "display_token_ids": display_token_ids,
        "generated_text": generated_text,
        "generated_tokens": len(completion_token_ids),
        "stopped_on_eos": stopped_on_eos,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a trained TinyStories checkpoint and sample text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--merges-path", type=Path, default=DEFAULT_MERGES_PATH)
    parser.add_argument("--special-token", type=str, default=SPECIAL_TOKEN)
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prefix to continue. Leave empty to start from <|endoftext|>.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=detect_default_device())
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--vocab-size", type=int, default=int(SECTION_72_MODEL_CONFIG["vocab_size"]))
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=int(SECTION_72_MODEL_CONFIG["d_model"]))
    parser.add_argument("--num-layers", type=int, default=int(SECTION_72_MODEL_CONFIG["num_layers"]))
    parser.add_argument("--num-heads", type=int, default=int(SECTION_72_MODEL_CONFIG["num_heads"]))
    parser.add_argument("--d-ff", type=int, default=int(SECTION_72_MODEL_CONFIG["d_ff"]))
    parser.add_argument("--rope-theta", type=float, default=float(SECTION_72_MODEL_CONFIG["rope_theta"]))
    return parser


if __name__ == "__main__":
    generate_text(**vars(build_parser().parse_args()))
