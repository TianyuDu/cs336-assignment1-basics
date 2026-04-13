"""Basic local training checks for CS336 Section 7.2.

This script supports:
1. Overfitting a single, fixed mini-batch (strong implementation sanity check).
2. A low-resource CPU/MPS training run tuned for 40,000,000 tokens.

Examples:
    uv run python scripts/721_basic_testing.py --mode overfit
    uv run python scripts/721_basic_testing.py --mode low-resource
    uv run python scripts/721_basic_testing.py --mode low-resource --batch-size 32 --context-length 256 --steps 5000
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data import data_loading
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.optimizer_adamw import AdamW
from cs336_basics.transformer_lm import TransformerLM

DEFAULT_TRAIN_TOKENS_PATH = (
    REPO_ROOT / "data" / "tokenizer_experiments" / "encoded" / "tinystories_10k" / "train.npy"
)
DEFAULT_VALID_TOKENS_PATH = (
    REPO_ROOT / "data" / "tokenizer_experiments" / "encoded" / "tinystories_10k" / "valid.npy"
)

LOW_RESOURCE_TARGET_TOKENS = 40_000_000

TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
TOKEN_DTYPES: dict[str, np.dtype] = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "int64": np.int64,
}


def detect_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def ensure_existing_path(path: Path, flag_name: str) -> Path:
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"{flag_name}={resolved} does not exist. Generate tokenized data first or pass a different path."
        )
    return resolved


def load_tokens(path: Path, token_dtype: str) -> np.ndarray:
    if path.suffix == ".npy":
        tokens = np.load(path, mmap_mode="r")
    else:
        tokens = np.memmap(path, dtype=TOKEN_DTYPES[token_dtype], mode="r")
    if not isinstance(tokens, np.ndarray) or tokens.ndim != 1:
        raise ValueError("Token path must point to a 1D numpy array or raw binary token file.")
    return tokens


def format_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def build_wandb_config(args: argparse.Namespace, mode_name: str) -> dict[str, object]:
    return {
        "mode": mode_name,
        "train_tokens_path": str(resolve_repo_path(args.train_tokens_path)),
        "valid_tokens_path": str(resolve_repo_path(args.valid_tokens_path)),
        "token_dtype": args.token_dtype,
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "dtype": args.dtype,
        "seed": args.seed,
        "device": args.device,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "eps": args.eps,
        "grad_clip": args.grad_clip,
        "grad_compile": bool(args.compile),
    }


def initialize_wandb_run(args: argparse.Namespace, mode_name: str):
    if not args.wandb:
        return None

    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(
            "W&B logging requested (--wandb), but wandb is not importable in this environment."
        ) from exc

    if args.wandb_mode in {"offline", "disabled"}:
        os.environ["WANDB_MODE"] = args.wandb_mode
    elif "WANDB_MODE" in os.environ:
        os.environ.pop("WANDB_MODE", None)

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        group=args.wandb_group,
        notes=args.wandb_notes,
        tags=args.wandb_tags,
        config=build_wandb_config(args, mode_name),
    )

    run.define_metric("global_step")
    for metric_pattern in ("overfit/*", "train/*", "val/*", "system/*"):
        run.define_metric(metric_pattern, step_metric="global_step")
    run.summary["run/mode"] = mode_name
    return run


def finalize_wandb_run(
    wandb_run,
    status: str = "completed",
    summary: dict[str, object] | None = None,
) -> None:
    if wandb_run is None:
        return
    if summary is not None:
        wandb_run.summary.update(summary)
    wandb_run.summary["run/status"] = status
    try:
        wandb_run.finish()
    except Exception:
        pass


def maybe_compile_model(model: torch.nn.Module, device: str, compile_model: bool) -> torch.nn.Module:
    if not compile_model or not hasattr(torch, "compile"):
        return model
    if device == "cpu":
        try:
            print("[compile] torch.compile(model) on CPU")
            return torch.compile(model)
        except Exception as exc:
            print(f"[compile] CPU compile failed, continuing without compile: {exc}")
            return model
    if device == "mps":
        try:
            print("[compile] torch.compile(model, backend=\"aot_eager\") on MPS")
            return torch.compile(model, backend="aot_eager")
        except Exception as exc:
            print(f"[compile] MPS compile failed, continuing without compile: {exc}")
            return model
    return model


def compute_batch_metrics(
    model: TransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        loss = float(cross_entropy(logits, targets).item())
        accuracy = float((logits.argmax(dim=-1) == targets).float().mean().item())
    if was_training:
        model.train()
    return loss, accuracy


def evaluate_validation(
    model: TransformerLM,
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_batches: int,
) -> tuple[float, float]:
    val_losses: list[float] = []
    val_accs: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(eval_batches):
            valid_inputs, valid_targets = data_loading(
                x=tokens,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            logits = model(valid_inputs)
            val_losses.append(float(cross_entropy(logits, valid_targets).item()))
            val_accs.append(float((logits.argmax(dim=-1) == valid_targets).float().mean().item()))
    model.train()
    return float(np.mean(val_losses)), float(np.mean(val_accs))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overfit a fixed mini-batch and run low-resource CPU/MPS checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["overfit", "low-resource"],
        default="overfit",
        help="overfit: train on one fixed batch; low-resource: train for a short full-loop run on CPU/MPS.",
    )
    parser.add_argument("--train-tokens-path", type=Path, default=DEFAULT_TRAIN_TOKENS_PATH)
    parser.add_argument("--valid-tokens-path", type=Path, default=DEFAULT_VALID_TOKENS_PATH)
    parser.add_argument(
        "--token-dtype",
        choices=sorted(TOKEN_DTYPES),
        default="uint16",
        help="Only used for raw binary token files, ignored for .npy files.",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--dtype", choices=sorted(TORCH_DTYPES), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps to run.")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="Token budget: used when --steps is omitted. On cpu/mps defaults to 40,000,000.",
    )

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--device", type=str, default=detect_default_device())
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable torch.compile optimization. If omitted, auto-enabled for CPU/MPS in low-resource mode, "
            "disabled otherwise."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Logging interval (steps).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=250,
        help="Validation-eval interval (steps) in low-resource mode.",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=5,
        help="Validation batches for low-resource mode.",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Early-stop threshold for validation loss (low-resource mode).",
    )
    parser.add_argument(
        "--target-val-loss",
        type=float,
        default=2.0,
        help="Target validation loss; default 2.0 for low-resource mode on CPU/MPS.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium", "default"],
        default=None,
        help=(
            "Optional torch.set_float32_matmul_precision override. This is skipped for MPS to avoid known TF32-related issues."
        ),
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable W&B logging for this run.",
    )
    parser.add_argument("--wandb-project", type=str, default="cs336-a1-local")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="721-basic-testing")
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    return parser


def configure_matmul_precision(device: str, precision: str | None) -> None:
    if precision is None:
        return
    if device == "mps":
        print(
            f"Skipping matmul precision override on MPS ({precision}) to avoid TF32 stability issues. "
            "Tip says not to force high on MPS."
        )
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(precision)
        print(f"Set float32 matmul precision to '{precision}'")


def run_overfit_mode(args: argparse.Namespace, train_tokens: np.ndarray, wandb_run=None) -> dict[str, object]:
    if args.steps is None:
        steps = 250
    else:
        steps = args.steps
    if steps <= 0:
        raise ValueError("--steps must be positive in overfit mode.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")

    if len(train_tokens) < args.context_length + 1:
        raise ValueError("Train token array is too short for the requested context length.")

    inputs, targets = data_loading(
        x=train_tokens,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
    )

    torch_dtype = TORCH_DTYPES[args.dtype]
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
        dtype=torch_dtype,
    )

    compile_model = args.compile if args.compile is not None else False
    model = maybe_compile_model(model, args.device, compile_model)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    initial_loss, initial_accuracy = compute_batch_metrics(model, inputs, targets)
    print("Single-batch overfit sanity check")
    print(f"  mode: overfit")
    print(f"  train tokens: {len(train_tokens):,}")
    print(f"  device: {args.device}")
    print(f"  dtype: {args.dtype}")
    print(f"  batch shape: {tuple(inputs.shape)}")
    print(f"  model params: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"  initial loss: {initial_loss:.4f}")
    print(f"  initial perplexity: {format_perplexity(initial_loss):.2f}")
    print(f"  initial token accuracy: {initial_accuracy:.3f}")
    if wandb_run is not None:
        wandb_run.log(
            {
                "global_step": 0,
                "overfit/initial_loss": initial_loss,
                "overfit/initial_perplexity": format_perplexity(initial_loss),
                "overfit/initial_accuracy": initial_accuracy,
            },
            step=0,
        )

    model.train()
    completed_steps = 0
    final_loss = initial_loss
    final_accuracy = initial_accuracy
    train_start = time.perf_counter()

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()
        completed_steps = step

        final_loss = float(loss.item())
        final_accuracy = float((logits.argmax(dim=-1) == targets).float().mean().item())
        if step == 1 or step == steps or step % args.log_every == 0:
            elapsed = time.perf_counter() - train_start
            print(
                f"[overfit step {step:04d}] loss={final_loss:.4f} "
                f"ppl={format_perplexity(final_loss):.2f} acc={final_accuracy:.3f} "
                f"elapsed={elapsed:.1f}s"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "global_step": step,
                        "overfit/train_loss": final_loss,
                        "overfit/train_perplexity": format_perplexity(final_loss),
                        "overfit/train_accuracy": final_accuracy,
                        "system/elapsed_sec": elapsed,
                        "system/steps_per_sec": step / max(elapsed, 1e-8),
                    },
                    step=step,
                )

    if args.stop_loss is not None and final_loss <= args.stop_loss:
        print(f"Reached overfit stop-loss threshold {args.stop_loss:.4f} at step {completed_steps}.")

    print("Overfit summary")
    print(f"  completed steps: {completed_steps}")
    print(f"  final loss: {final_loss:.4f}")
    print(f"  final perplexity: {format_perplexity(final_loss):.2f}")
    print(f"  final token accuracy: {final_accuracy:.3f}")
    print(f"  loss improvement: {initial_loss - final_loss:.4f}")
    return {
        "final_loss": final_loss,
        "final_perplexity": format_perplexity(final_loss),
        "final_accuracy": final_accuracy,
        "loss_improvement": initial_loss - final_loss,
        "completed_steps": completed_steps,
    }


def run_low_resource_mode(
    args: argparse.Namespace,
    train_tokens: np.ndarray,
    valid_tokens: np.ndarray,
    wandb_run=None,
) -> dict[str, object]:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive.")
    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be positive.")

    if len(train_tokens) < args.context_length + 1:
        raise ValueError("Train token array is too short for the requested context length.")
    if len(valid_tokens) < args.context_length + 1:
        raise ValueError("Valid token array is too short for the requested context length.")

    tokens_per_step = args.batch_size * args.context_length
    if args.steps is not None:
        max_iters = args.steps
        target_tokens = args.steps * tokens_per_step
    else:
        if args.target_tokens is not None:
            target_tokens = args.target_tokens
        elif args.device in {"cpu", "mps"}:
            target_tokens = LOW_RESOURCE_TARGET_TOKENS
        else:
            target_tokens = LOW_RESOURCE_TARGET_TOKENS
        max_iters = max(1, math.ceil(target_tokens / tokens_per_step))

    if args.warmup_iters is not None:
        warmup_iters = args.warmup_iters
    else:
        warmup_iters = max(0, int(round(max_iters * args.warmup_fraction)))
    warmup_iters = min(max(warmup_iters, 0), max(max_iters - 1, 0))

    min_lr = args.min_lr if args.min_lr is not None else args.lr * 0.1

    if not args.device.startswith("mps"):
        configure_matmul_precision(args.device, args.matmul_precision)
    else:
        # Per low-resource note: do not force torch.set_float32_matmul_precision("high") for MPS.
        if args.matmul_precision is not None:
            print(
                f"Ignoring matmul precision '{args.matmul_precision}' on MPS to avoid TF32 backend issues."
            )

    torch_dtype = TORCH_DTYPES[args.dtype]
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
        dtype=torch_dtype,
    )

    compile_model = args.compile if args.compile is not None else args.device in {"cpu", "mps"}
    model = maybe_compile_model(model, args.device, compile_model)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    if args.mode == "low-resource":
        print("Low-resource CPU/MPS training check")
        print(f"  target tokens requested: {target_tokens:,}")
        print(f"  batch size: {args.batch_size}")
        print(f"  context length: {args.context_length}")
        print(f"  computed steps: {max_iters}")
        print(f"  lr max/min: {args.lr:.2e}/{min_lr:.2e}")
        print(f"  warmup iterations: {warmup_iters}")
        print(f"  cosine schedule end step: {max_iters}")

    model.train()
    train_start = time.perf_counter()
    final_train_loss = float("nan")
    final_val_loss = float("nan")
    final_val_accuracy = float("nan")
    completed_steps = 0

    for step in range(1, max_iters + 1):
        current_lr = get_lr_cosine_schedule(
            t=step,
            alpha_max=args.lr,
            alpha_min=min_lr,
            T_w=warmup_iters,
            T_c=max_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        inputs, targets = data_loading(
            x=train_tokens,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()
        completed_steps = step

        final_train_loss = float(loss.item())
        if step == 1 or step % args.log_every == 0 or step == max_iters:
            train_accuracy = float((logits.argmax(dim=-1) == targets).float().mean().item())
            elapsed = time.perf_counter() - train_start
            tokens_seen = step * tokens_per_step
            tokens_per_sec = tokens_seen / max(elapsed, 1e-8)
            print(
                f"[low-resource step {step:05d}] lr={current_lr:.2e} train_loss={final_train_loss:.4f} "
                f"train_acc={train_accuracy:.3f} tok/s={tokens_per_sec:.1f} elapsed={elapsed:.1f}s"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "global_step": step,
                        "train/lr": current_lr,
                        "train/loss": final_train_loss,
                        "train/perplexity": format_perplexity(final_train_loss),
                        "train/accuracy": train_accuracy,
                        "system/tokens_per_sec": tokens_per_sec,
                        "system/elapsed_sec": elapsed,
                    },
                    step=step,
                )

        if step % args.eval_every == 0 or step == max_iters:
            val_loss, val_accuracy = evaluate_validation(
                model=model,
                tokens=valid_tokens,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                eval_batches=args.eval_batches,
            )
            final_val_loss = val_loss
            final_val_accuracy = val_accuracy
            elapsed = time.perf_counter() - train_start
            print(
                f"[low-resource step {step:05d}] val_loss={val_loss:.4f} val_ppl={format_perplexity(val_loss):.2f} "
                f"val_acc={val_accuracy:.3f} elapsed={elapsed:.1f}s"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "global_step": step,
                        "val/loss": val_loss,
                        "val/perplexity": format_perplexity(val_loss),
                        "val/accuracy": val_accuracy,
                    },
                    step=step,
                )
            if args.target_val_loss is not None and val_loss <= args.target_val_loss:
                print(
                    f"Reached target validation loss {args.target_val_loss:.4f} at step {step}. "
                    "Stopping early for low-resource check."
                )
                break
            if args.stop_loss is not None and final_val_loss <= args.stop_loss:
                print(f"Reached explicit stop-loss threshold {args.stop_loss:.4f} at step {step}.")
                break

    elapsed_total = time.perf_counter() - train_start
    tokens_processed = completed_steps * tokens_per_step
    print("Low-resource summary")
    print(f"  completed steps: {completed_steps}")
    print(f"  target steps: {max_iters}")
    print(f"  tokens requested: {target_tokens:,}")
    print(f"  tokens processed: {tokens_processed:,}")
    print(f"  elapsed: {elapsed_total:.1f}s")
    print(f"  final train loss: {final_train_loss:.4f}")
    print(f"  final val loss: {final_val_loss:.4f}")
    if not math.isnan(final_val_loss):
        print(f"  final val perplexity: {format_perplexity(final_val_loss):.2f}")
        print(f"  final val accuracy: {final_val_accuracy:.3f}")
    return {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_val_accuracy": final_val_accuracy,
        "target_tokens": target_tokens,
        "steps_completed": completed_steps,
        "target_steps": max_iters,
        "total_elapsed_sec": elapsed_total,
        "tokens_processed": tokens_processed,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.compile is None:
        args.compile = args.device in {"cpu", "mps"} and args.mode == "low-resource"

    train_tokens_path = ensure_existing_path(args.train_tokens_path, "--train-tokens-path")
    valid_tokens_path = ensure_existing_path(args.valid_tokens_path, "--valid-tokens-path")

    train_tokens = load_tokens(train_tokens_path, args.token_dtype)
    valid_tokens = load_tokens(valid_tokens_path, args.token_dtype)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested --device cuda but CUDA is not available in this environment.")
    if args.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("Requested --device mps but MPS is not available in this environment.")

    wandb_run = initialize_wandb_run(args, args.mode)
    try:
        if args.mode == "low-resource":
            final_summary = run_low_resource_mode(
                args=args,
                train_tokens=train_tokens,
                valid_tokens=valid_tokens,
                wandb_run=wandb_run,
            )
        else:
            final_summary = run_overfit_mode(args=args, train_tokens=train_tokens, wandb_run=wandb_run)
        if wandb_run is not None:
            finalize_wandb_run(wandb_run, status="completed", summary=final_summary)
    except BaseException as exc:
        if wandb_run is not None:
            try:
                wandb_run.summary["run/error"] = repr(exc)
            except Exception:
                pass
            finalize_wandb_run(wandb_run, status="failed")
        raise


if __name__ == "__main__":
    main()
