"""Launch the shared Section 7.2 training pipeline for a learning-rate sweep.

Examples:
  uv run python scripts/723_learning_rate_tuning.py --local --steps 100
  modal run scripts/723_learning_rate_tuning.py
"""

from __future__ import annotations

import argparse
import math
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

DEFAULT_LEARNING_RATES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
DEFAULT_TARGET_TOKENS = 327_680_000


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200", max_containers=3)
def launch_training_trial(train_kwargs: dict[str, Any]) -> dict[str, Any]:
    return train(**train_kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a shared-pipeline learning-rate sweep for the Section 7.2 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--learning-rates",
        nargs="*",
        type=float,
        default=DEFAULT_LEARNING_RATES,
        help="Learning-rate values to try.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run each training job locally instead of submitting it to Modal.",
    )
    parser.add_argument("--train-tokens-path", type=Path, default=DEFAULT_REMOTE_TRAIN_PATH)
    parser.add_argument("--valid-tokens-path", type=Path, default=DEFAULT_REMOTE_VALID_PATH)
    parser.add_argument("--token-dtype", type=str, default=str(SECTION_72_MODEL_CONFIG["token_dtype"]))
    parser.add_argument("--vocab-size", type=int, default=int(SECTION_72_MODEL_CONFIG["vocab_size"]))
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=detect_default_device())
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--wandb-project", type=str, default="cs336-a1")
    parser.add_argument("--wandb-run-prefix", type=str, default="723-lr-sweep")
    parser.add_argument("--wandb-group", type=str, default="723-learning-rate-tuning")
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    return parser


def build_trial_kwargs(args: argparse.Namespace, steps: int) -> list[dict[str, Any]]:
    is_local = args.local
    train_tokens_path = resolve_execution_path(args.train_tokens_path, local=is_local)
    valid_tokens_path = resolve_execution_path(args.valid_tokens_path, local=is_local)
    device = args.device if is_local else "cuda"
    warmup_iters = max(0, int(round(steps * args.warmup_fraction)))
    cosine_cycle_iters = max(steps, warmup_iters + 1)

    trial_kwargs: list[dict[str, Any]] = []
    for learning_rate in args.learning_rates:
        lr_slug = f"{learning_rate:.6g}".replace(".", "_")
        trial_kwargs.append(
            {
                **SECTION_72_MODEL_CONFIG,
                "train_tokens_path": str(train_tokens_path),
                "valid_tokens_path": str(valid_tokens_path),
                "token_dtype": args.token_dtype,
                "vocab_size": args.vocab_size,
                "context_length": args.context_length,
                "batch_size": args.batch_size,
                "max_iters": steps,
                "max_lr": learning_rate,
                "min_lr": learning_rate * args.min_lr_ratio,
                "warmup_iters": warmup_iters,
                "cosine_cycle_iters": cosine_cycle_iters,
                "log_every": args.log_every,
                "eval_every": args.eval_every,
                "eval_batches": args.eval_batches,
                "save_every": steps,
                "checkpoint_path": str(
                    checkpoint_path(
                        local=is_local,
                        subdir="723-learning-rate-tuning",
                        filename=f"{lr_slug}.pt",
                    )
                ),
                "seed": args.seed,
                "device": device,
                "dtype": args.dtype,
                "wandb_project": args.wandb_project,
                "wandb_run_name": f"{args.wandb_run_prefix}-lr-{lr_slug}",
                "wandb_group": args.wandb_group,
                "wandb_job_type": "lr-sweep",
                "wandb_notes": args.wandb_notes,
                "wandb_tags": args.wandb_tags,
                "wandb_mode": args.wandb_mode,
            }
        )
    return trial_kwargs


def steps_for_run(args: argparse.Namespace) -> int:
    if args.steps is not None:
        if args.steps <= 0:
            raise ValueError("--steps must be positive.")
        return args.steps
    return steps_from_token_budget(
        batch_size=args.batch_size,
        context_length=args.context_length,
        target_tokens=args.target_tokens,
    )


def print_launch_summary(args: argparse.Namespace, steps: int) -> None:
    total_tokens = steps * args.batch_size * args.context_length
    print("Learning-rate sweep")
    print(f"  execution: {'local' if args.local else 'modal'}")
    print(f"  learning rates: {args.learning_rates}")
    print(f"  steps: {steps}")
    print(f"  approx tokens per trial: {total_tokens}")


def print_trial_summary(learning_rate: float, summary: dict[str, Any]) -> None:
    print(f"\nLearning rate {learning_rate:.6g}")
    print(f"  status: {summary['status']}")
    if summary.get("final_train_loss") is not None:
        print(f"  final train loss: {summary['final_train_loss']:.4f}")
    if summary.get("final_val_loss") is not None:
        print(f"  final val loss: {summary['final_val_loss']:.4f}")
    if summary.get("best_val_loss") is not None:
        print(f"  best val loss: {summary['best_val_loss']:.4f}")


def choose_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    def result_score(item: dict[str, Any]) -> float:
        final_val_loss = item.get("final_val_loss")
        if final_val_loss is not None:
            return float(final_val_loss)
        best_val_loss = item.get("best_val_loss")
        if best_val_loss is not None:
            return float(best_val_loss)
        return math.inf

    return min(results, key=result_score)


def run_from_cli() -> None:
    args = build_parser().parse_args()
    steps = steps_for_run(args)
    trial_kwargs = build_trial_kwargs(args, steps)
    print_launch_summary(args, steps)

    results: list[dict[str, Any]] = []
    if args.local:
        for learning_rate, train_kwargs in zip(args.learning_rates, trial_kwargs, strict=False):
            summary = launch_training_trial.local(train_kwargs)
            summary["learning_rate"] = learning_rate
            print_trial_summary(learning_rate, summary)
            results.append(summary)
    else:
        for learning_rate, summary in zip(
            args.learning_rates,
            launch_training_trial.map(trial_kwargs),
            strict=False,
        ):
            summary["learning_rate"] = learning_rate
            print_trial_summary(learning_rate, summary)
            results.append(summary)

    if not results:
        print("No results to summarize.")
        return

    best_result = choose_best_result(results)
    print("\nSweep complete")
    print(
        f"Best learning rate: {best_result['learning_rate']} "
        f"(final_val_loss={best_result.get('final_val_loss')})"
    )


@app.local_entrypoint()
def modal_main() -> None:
    run_from_cli()


if __name__ == "__main__":
    run_from_cli()
