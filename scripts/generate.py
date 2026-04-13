"""Load a TinyStories checkpoint and sample text from it.

Usage:
  uv run python scripts/generate.py
  uv run python scripts/generate.py --prompt "Once upon a time"
  uv run python scripts/generate.py --temperature 0.7 --top-p 0.9
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs336_basics.decoding import decode
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training_launcher_utils import SECTION_72_MODEL_CONFIG, detect_default_device
from cs336_basics.transformer_lm import TransformerLM

# Defaults -- change these to point at your own checkpoint / tokenizer.
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "723-learning-rate-tuning" / "0_001.pt"
VOCAB_PATH = REPO_ROOT / "data" / "bpe_tinystories" / "vocab.json"
MERGES_PATH = REPO_ROOT / "data" / "bpe_tinystories" / "merges.txt"
EOT = "<|endoftext|>"

DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--vocab-path", type=Path, default=VOCAB_PATH)
    parser.add_argument("--merges-path", type=Path, default=MERGES_PATH)
    parser.add_argument("--prompt", type=str, default="", help="Text to continue. Empty = fresh sample.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=detect_default_device())
    parser.add_argument("--dtype", choices=DTYPE_MAP.keys(), default="float32")
    parser.add_argument("--context-length", type=int, default=256)
    args = parser.parse_args()

    # Model architecture from the Section 7.2 config.
    cfg = SECTION_72_MODEL_CONFIG

    # Seed for reproducible sampling.
    torch.manual_seed(args.seed)

    # --- Step 1: Load tokenizer ---
    print(f"Loading tokenizer from {args.vocab_path} ...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(args.vocab_path),
        merges_filepath=str(args.merges_path),
        special_tokens=[EOT],
    )
    eos_id = tokenizer.special_token_to_id[EOT]

    # If no prompt is given, feed the model the EOT token so it starts a fresh story.
    prompt_ids = tokenizer.encode(args.prompt) if args.prompt else [eos_id]

    # --- Step 2: Build model and load weights ---
    device = torch.device(args.device)
    print(f"Building model on {device} ...")
    model = TransformerLM(
        vocab_size=int(cfg["vocab_size"]),
        context_length=args.context_length,
        d_model=int(cfg["d_model"]),
        num_layers=int(cfg["num_layers"]),
        num_heads=int(cfg["num_heads"]),
        d_ff=int(cfg["d_ff"]),
        rope_theta=float(cfg["rope_theta"]),
        device=device,
        dtype=DTYPE_MAP[args.dtype],
    )

    print(f"Loading checkpoint from {args.checkpoint_path} ...")
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    # Strip _orig_mod. prefix that torch.compile adds.
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # --- Step 3: Generate ---
    print(f"Generating up to {args.max_new_tokens} tokens (temp={args.temperature}, top_p={args.top_p}) ...")
    all_ids = decode(
        model=model,
        prompt_token_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_id,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # --- Step 4: Decode and print ---
    new_ids = all_ids[len(prompt_ids):]
    stopped_on_eos = bool(new_ids and new_ids[-1] == eos_id)

    # For display: hide the synthetic EOT start token and any trailing EOT.
    display_ids = list(all_ids)
    if not args.prompt:
        display_ids = display_ids[1:]  # drop leading EOT
    if stopped_on_eos and display_ids:
        display_ids = display_ids[:-1]  # drop trailing EOT

    text = tokenizer.decode(display_ids)

    print(f"\n--- Config ---")
    print(f"checkpoint : {args.checkpoint_path}")
    if "iteration" in ckpt:
        print(f"iteration  : {ckpt['iteration']}")
    print(f"device     : {device}")
    print(f"dtype      : {args.dtype}")
    print(f"temperature: {args.temperature}")
    print(f"top_p      : {args.top_p}")
    print(f"seed       : {args.seed}")
    print(f"prompt     : {args.prompt!r}" if args.prompt else f"prompt     : (none, fresh sample)")
    print(f"new tokens : {len(new_ids)}")
    print(f"stopped_eos: {stopped_on_eos}")
    print(f"\n--- Generated text ---\n")
    print(text)
