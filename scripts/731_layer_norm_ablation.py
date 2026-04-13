"""Ablation 1: Compare training with and without RMSNorm across learning rates.

For each learning rate, launches two trials: one with RMSNorm (baseline) and
one without (ablation). This makes it easy to compare stability and final loss.

Examples:
  uv run python scripts/731_layer_norm_ablation.py --local --steps 100
  modal run scripts/731_layer_norm_ablation.py
  modal run scripts/731_layer_norm_ablation.py --learning-rates "1e-3,5e-4,1e-4,5e-5,1e-5"
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import modal

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

# The optimal LR from section 7.2.3, plus lower candidates to probe stability.
DEFAULT_LEARNING_RATES = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
DEFAULT_TARGET_TOKENS = 327_680_000


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    max_containers=5,
    timeout=7200,
    secrets=[modal.Secret.from_name("wandb")],
)
def launch_training_trial(train_kwargs: dict[str, Any]) -> dict[str, Any]:
    return train(**train_kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablation 1: train with vs without RMSNorm across learning rates.",
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
    parser.add_argument("--wandb-run-prefix", type=str, default="731")
    parser.add_argument("--wandb-group", type=str, default="731-layer-norm-ablation")
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return parser


def build_trial_kwargs(args: argparse.Namespace, steps: int) -> list[dict[str, Any]]:
    """Build trial specs for every (learning_rate, rmsnorm_enabled) combination."""
    is_local = args.local
    train_tokens_path = resolve_execution_path(args.train_tokens_path, local=is_local)
    valid_tokens_path = resolve_execution_path(args.valid_tokens_path, local=is_local)
    device = args.device if is_local else "cuda"
    warmup_iters = max(0, int(round(steps * args.warmup_fraction)))
    cosine_cycle_iters = max(steps, warmup_iters + 1)

    trial_kwargs: list[dict[str, Any]] = []
    for learning_rate in args.learning_rates:
        lr_slug = f"{learning_rate:.6g}".replace(".", "_")
        for disable_rmsnorm in [False, True]:
            norm_tag = "no-rmsnorm" if disable_rmsnorm else "with-rmsnorm"
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
                            subdir="731-layer-norm-ablation",
                            filename=f"{norm_tag}-lr-{lr_slug}.pt",
                        )
                    ),
                    "seed": args.seed,
                    "device": device,
                    "dtype": args.dtype,
                    "disable_rmsnorm": disable_rmsnorm,
                    "wandb_project": args.wandb_project,
                    "wandb_run_name": f"{args.wandb_run_prefix}-{norm_tag}-lr-{lr_slug}",
                    "wandb_group": args.wandb_group,
                    "wandb_job_type": "layer-norm-ablation",
                    "wandb_notes": args.wandb_notes,
                    "wandb_tags": [
                        "731-layer-norm-ablation",
                        norm_tag,
                        f"lr-{lr_slug}",
                        *(args.wandb_tags or []),
                    ],
                    "wandb_mode": args.wandb_mode,
                    # Stash metadata for summary printing.
                    "_meta_lr": learning_rate,
                    "_meta_norm_tag": norm_tag,
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


def print_launch_summary(args: argparse.Namespace, steps: int, num_trials: int) -> None:
    total_tokens = steps * args.batch_size * args.context_length
    print("Layer-norm ablation (with vs without RMSNorm)")
    print(f"  execution: {'local' if args.local else 'modal'}")
    print(f"  learning rates: {args.learning_rates}")
    print(f"  trials per LR: 2 (with-rmsnorm, no-rmsnorm)")
    print(f"  total trials: {num_trials}")
    print(f"  steps per trial: {steps}")
    print(f"  approx tokens per trial: {total_tokens}")


def print_trial_summary(label: str, summary: dict[str, Any]) -> None:
    print(f"\n{label}")
    print(f"  status: {summary['status']}")
    if summary.get("final_train_loss") is not None:
        print(f"  final train loss: {summary['final_train_loss']:.4f}")
    if summary.get("final_val_loss") is not None:
        print(f"  final val loss: {summary['final_val_loss']:.4f}")
    if summary.get("best_val_loss") is not None:
        print(f"  best val loss: {summary['best_val_loss']:.4f}")


def _result_score(item: dict[str, Any]) -> float:
    final_val_loss = item.get("final_val_loss")
    if final_val_loss is not None and math.isfinite(final_val_loss):
        return float(final_val_loss)
    best_val_loss = item.get("best_val_loss")
    if best_val_loss is not None and math.isfinite(best_val_loss):
        return float(best_val_loss)
    return math.inf


def _parse_csv_floats(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("learning_rates must include at least one comma-separated float.")
    return [float(item) for item in items]


def _parse_csv_strings(value: str) -> list[str] | None:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _strip_meta(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove internal _meta_* keys before passing to train()."""
    return {k: v for k, v in kwargs.items() if not k.startswith("_meta_")}


def run_from_args(args: argparse.Namespace) -> None:
    steps = steps_for_run(args)
    all_trial_kwargs = build_trial_kwargs(args, steps)
    print_launch_summary(args, steps, len(all_trial_kwargs))

    results: list[dict[str, Any]] = []
    clean_kwargs = [_strip_meta(tk) for tk in all_trial_kwargs]

    if args.local:
        for meta, train_kwargs in zip(all_trial_kwargs, clean_kwargs, strict=False):
            summary = launch_training_trial.local(train_kwargs)
            summary["learning_rate"] = meta["_meta_lr"]
            summary["norm_tag"] = meta["_meta_norm_tag"]
            label = f"LR {meta['_meta_lr']:.6g} | {meta['_meta_norm_tag']}"
            print_trial_summary(label, summary)
            results.append(summary)
    else:
        for meta, summary in zip(
            all_trial_kwargs,
            launch_training_trial.map(clean_kwargs),
            strict=False,
        ):
            summary["learning_rate"] = meta["_meta_lr"]
            summary["norm_tag"] = meta["_meta_norm_tag"]
            label = f"LR {meta['_meta_lr']:.6g} | {meta['_meta_norm_tag']}"
            print_trial_summary(label, summary)
            results.append(summary)

    if not results:
        print("No results to summarize.")
        return

    # Summarise by group.
    with_norm = [r for r in results if r["norm_tag"] == "with-rmsnorm"]
    no_norm = [r for r in results if r["norm_tag"] == "no-rmsnorm"]

    print("\n===== Summary =====")
    if with_norm:
        best_with = min(with_norm, key=_result_score)
        print(
            f"Best with RMSNorm:    LR={best_with['learning_rate']:.6g}  "
            f"final_val_loss={best_with.get('final_val_loss')}"
        )
    if no_norm:
        best_without = min(no_norm, key=_result_score)
        print(
            f"Best without RMSNorm: LR={best_without['learning_rate']:.6g}  "
            f"final_val_loss={best_without.get('final_val_loss')}"
        )


@app.local_entrypoint()
def modal_main(
    learning_rates: str = "1e-3,5e-4,1e-4,5e-5,1e-5",
    train_tokens_path: str = str(DEFAULT_REMOTE_TRAIN_PATH),
    valid_tokens_path: str = str(DEFAULT_REMOTE_VALID_PATH),
    token_dtype: str = str(SECTION_72_MODEL_CONFIG["token_dtype"]),
    vocab_size: int = int(SECTION_72_MODEL_CONFIG["vocab_size"]),
    context_length: int = 256,
    batch_size: int = 32,
    steps: int = 0,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    min_lr_ratio: float = 0.1,
    warmup_fraction: float = 0.02,
    log_every: int = 250,
    eval_every: int = 250,
    eval_batches: int = 4,
    seed: int = 0,
    dtype: str = "float32",
    wandb_project: str = "cs336-a1",
    wandb_run_prefix: str = "731",
    wandb_group: str = "731-layer-norm-ablation",
    wandb_notes: str = "",
    wandb_tags: str = "",
    wandb_mode: str = "online",
) -> None:
    args = argparse.Namespace(
        learning_rates=_parse_csv_floats(learning_rates),
        local=False,
        train_tokens_path=Path(train_tokens_path),
        valid_tokens_path=Path(valid_tokens_path),
        token_dtype=token_dtype,
        vocab_size=vocab_size,
        context_length=context_length,
        batch_size=batch_size,
        steps=steps if steps > 0 else None,
        target_tokens=target_tokens,
        min_lr_ratio=min_lr_ratio,
        warmup_fraction=warmup_fraction,
        log_every=log_every,
        eval_every=eval_every,
        eval_batches=eval_batches,
        seed=seed,
        device="cuda",
        dtype=dtype,
        wandb_project=wandb_project,
        wandb_run_prefix=wandb_run_prefix,
        wandb_group=wandb_group,
        wandb_notes=wandb_notes or None,
        wandb_tags=_parse_csv_strings(wandb_tags),
        wandb_mode=wandb_mode,
    )
    run_from_args(args)


def run_from_cli() -> None:
    args = build_parser().parse_args()
    run_from_args(args)


if __name__ == "__main__":
    run_from_cli()
