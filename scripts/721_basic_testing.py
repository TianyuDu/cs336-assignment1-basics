"""Launch the shared Section 7.2 training pipeline for basic checks.

Examples:
  uv run python scripts/721_basic_testing.py --mode overfit --local
  uv run python scripts/721_basic_testing.py --mode low-resource --local --steps 500
  modal run scripts/721_basic_testing.py --mode low-resource
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs336_basics.modal_utils import VOLUME_MOUNTS, app, build_image
from cs336_basics.training_launcher_utils import (
    DEFAULT_REMOTE_TRAIN_PATH,
    DEFAULT_REMOTE_VALID_PATH,
    SECTION_72_MODEL_CONFIG,
    checkpoint_path,
    detect_default_device,
    resolve_execution_path,
    steps_from_token_budget,
)
from cs336_basics.training_together import train

LOW_RESOURCE_TARGET_TOKENS = 40_000_000


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200")
def launch_training(train_kwargs: dict[str, Any]) -> dict[str, Any]:
    return train(**train_kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the shared training pipeline for overfit and low-resource checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["overfit", "low-resource"], default="overfit")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run the training job locally instead of submitting it to Modal.",
    )
    parser.add_argument("--train-tokens-path", type=Path, default=DEFAULT_REMOTE_TRAIN_PATH)
    parser.add_argument("--valid-tokens-path", type=Path, default=DEFAULT_REMOTE_VALID_PATH)
    parser.add_argument("--token-dtype", type=str, default=str(SECTION_72_MODEL_CONFIG["token_dtype"]))
    parser.add_argument("--vocab-size", type=int, default=int(SECTION_72_MODEL_CONFIG["vocab_size"]))
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=LOW_RESOURCE_TARGET_TOKENS)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--target-val-loss", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=detect_default_device())
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--wandb-project", type=str, default="cs336-a1-local")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="721-basic-testing")
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    return parser


def build_train_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    is_local = args.local
    train_tokens_path = resolve_execution_path(args.train_tokens_path, local=is_local)
    valid_tokens_path = resolve_execution_path(args.valid_tokens_path, local=is_local)
    device = args.device if is_local else "cuda"

    if args.mode == "overfit":
        max_iters = args.steps if args.steps is not None else 250
        min_lr = args.lr
        warmup_iters = 0
        eval_every = max_iters
        eval_batches = 1
        target_val_loss = None
        fixed_batch = True
        valid_tokens_path = train_tokens_path
        checkpoint = checkpoint_path(
            local=is_local,
            subdir="721-basic-testing",
            filename="overfit.pt",
        )
        run_name = args.wandb_run_name or "721-overfit"
        job_type = "overfit"
    else:
        max_iters = args.steps
        if max_iters is None:
            max_iters = steps_from_token_budget(
                batch_size=args.batch_size,
                context_length=args.context_length,
                target_tokens=args.target_tokens,
            )
        min_lr = args.min_lr if args.min_lr is not None else args.lr * 0.1
        warmup_iters = max(0, int(round(max_iters * args.warmup_fraction)))
        eval_every = args.eval_every
        eval_batches = args.eval_batches
        target_val_loss = args.target_val_loss
        fixed_batch = False
        checkpoint = checkpoint_path(
            local=is_local,
            subdir="721-basic-testing",
            filename="low-resource.pt",
        )
        run_name = args.wandb_run_name or "721-low-resource"
        job_type = "low-resource"

    cosine_cycle_iters = max(max_iters, warmup_iters + 1)
    return {
        **SECTION_72_MODEL_CONFIG,
        "train_tokens_path": str(train_tokens_path),
        "valid_tokens_path": str(valid_tokens_path),
        "token_dtype": args.token_dtype,
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "max_iters": max_iters,
        "max_lr": args.lr,
        "min_lr": min_lr,
        "warmup_iters": warmup_iters,
        "cosine_cycle_iters": cosine_cycle_iters,
        "log_every": args.log_every,
        "eval_every": eval_every,
        "eval_batches": eval_batches,
        "save_every": max_iters,
        "checkpoint_path": str(checkpoint),
        "seed": args.seed,
        "device": device,
        "dtype": args.dtype,
        "fixed_batch": fixed_batch,
        "target_val_loss": target_val_loss,
        "wandb_project": args.wandb_project,
        "wandb_run_name": run_name,
        "wandb_group": args.wandb_group,
        "wandb_job_type": job_type,
        "wandb_notes": args.wandb_notes,
        "wandb_tags": args.wandb_tags,
        "wandb_mode": args.wandb_mode,
    }


def print_launch_summary(args: argparse.Namespace, train_kwargs: dict[str, Any]) -> None:
    print("Basic testing launcher")
    print(f"  mode: {args.mode}")
    print(f"  execution: {'local' if args.local else 'modal'}")
    print(f"  train tokens: {train_kwargs['train_tokens_path']}")
    print(f"  valid tokens: {train_kwargs['valid_tokens_path']}")
    print(f"  device: {train_kwargs['device']}")
    print(f"  steps: {train_kwargs['max_iters']}")


def print_result_summary(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    print("\nRun complete")
    print(f"  status: {summary['status']}")
    print(f"  final step: {summary['final_step']}")
    print(f"  checkpoint: {summary['checkpoint_path']}")
    if summary.get("final_train_loss") is not None:
        print(f"  final train loss: {summary['final_train_loss']:.4f}")
    if summary.get("final_train_perplexity") is not None:
        print(f"  final train perplexity: {summary['final_train_perplexity']:.2f}")
    if summary.get("final_train_accuracy") is not None:
        print(f"  final train accuracy: {summary['final_train_accuracy']:.3f}")
    if args.mode == "low-resource" and summary.get("final_val_loss") is not None:
        print(f"  final val loss: {summary['final_val_loss']:.4f}")
    if args.mode == "low-resource" and summary.get("best_val_loss") is not None:
        print(f"  best val loss: {summary['best_val_loss']:.4f} at step {summary['best_val_step']}")


def run_from_cli() -> None:
    args = build_parser().parse_args()
    train_kwargs = build_train_kwargs(args)
    print_launch_summary(args, train_kwargs)
    if args.local:
        summary = launch_training.local(train_kwargs)
    else:
        summary = launch_training.remote(train_kwargs)
    print_result_summary(args, summary)


@app.local_entrypoint()
def modal_main() -> None:
    run_from_cli()


if __name__ == "__main__":
    run_from_cli()
