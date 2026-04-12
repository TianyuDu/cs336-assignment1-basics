import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import wandb

# local implementations.
from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data import data_loading
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.optimizer_adamw import AdamW
from cs336_basics.transformer_lm import TransformerLM


def train(
    *,
    train_tokens_path: str,
    valid_tokens_path: str,
    token_dtype: str = "uint16",
    vocab_size: int,
    context_length: int = 256,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    d_ff: int = 2048,
    rope_theta: float = 10000.0,
    batch_size: int = 16,
    max_iters: int = 2000,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_iters: int = 100,
    cosine_cycle_iters: int = 2000,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    grad_clip: float = 1.0,
    log_every: int = 10,
    eval_every: int = 100,
    eval_batches: int = 10,
    save_every: int = 500,
    checkpoint_path: str = "checkpoints/latest.pt",
    resume_from: str | None = None,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "float32",
    wandb_project: str = "cs336-a1",
    wandb_run_name: str | None = None,
) -> None:
    if max_iters <= 0:
        raise ValueError("--max-iters must be positive.")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if context_length <= 0:
        raise ValueError("--context-length must be positive.")
    if eval_batches <= 0:
        raise ValueError("--eval-batches must be positive.")
    if log_every <= 0 or eval_every <= 0 or save_every <= 0:
        raise ValueError("--log-every, --eval-every, and --save-every must all be positive.")
    if cosine_cycle_iters <= warmup_iters:
        raise ValueError("--cosine-cycle-iters must be greater than --warmup-iters.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device but CUDA is not available.")

    # Make sampling and initialization reproducible.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch_dtype_map: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    token_dtype_map: dict[str, np.dtype] = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int32": np.int32,
        "int64": np.int64,
    }
    model_dtype = torch_dtype_map[dtype]
    token_dtype_np = token_dtype_map[token_dtype]

    # Load token arrays lazily from disk with memory mapping.
    train_path = Path(train_tokens_path)
    valid_path = Path(valid_tokens_path)
    if train_path.suffix == ".npy":
        train_tokens = np.load(train_path, mmap_mode="r")
    else:
        train_tokens = np.memmap(train_path, dtype=token_dtype_np, mode="r")
    if valid_path.suffix == ".npy":
        valid_tokens = np.load(valid_path, mmap_mode="r")
    else:
        valid_tokens = np.memmap(valid_path, dtype=token_dtype_np, mode="r")

    if not isinstance(train_tokens, np.ndarray) or not isinstance(valid_tokens, np.ndarray):
        raise ValueError("Token paths must point to numpy arrays or raw binary token files.")
    if train_tokens.ndim != 1 or valid_tokens.ndim != 1:
        raise ValueError("Both train and validation token arrays must be 1D.")
    if len(train_tokens) < context_length + 1:
        raise ValueError("Train token array is too short for the requested context length.")
    if len(valid_tokens) < context_length + 1:
        raise ValueError("Validation token array is too short for the requested context length.")

    # Always save to a user-provided checkpoint path (created if missing).
    checkpoint_path_obj = Path(checkpoint_path)
    checkpoint_path_obj.parent.mkdir(parents=True, exist_ok=True)
    run_config = {
        "train_tokens_path": train_tokens_path,
        "valid_tokens_path": valid_tokens_path,
        "token_dtype": token_dtype,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "rope_theta": rope_theta,
        "batch_size": batch_size,
        "max_iters": max_iters,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "warmup_iters": warmup_iters,
        "cosine_cycle_iters": cosine_cycle_iters,
        "weight_decay": weight_decay,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "grad_clip": grad_clip,
        "log_every": log_every,
        "eval_every": eval_every,
        "eval_batches": eval_batches,
        "save_every": save_every,
        "checkpoint_path": checkpoint_path,
        "resume_from": resume_from,
        "seed": seed,
        "device": device,
        "dtype": dtype,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
    }
    # Keep W&B setup direct and visible for students: one init, periodic logs, one finish.
    run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=run_config,
    )

    # Build model + optimizer once, then train in a single readable loop.
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=torch.device(device),
        dtype=model_dtype,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )

    start_step = 0
    if resume_from:
        start_step = load_checkpoint(resume_from, model, optimizer)
        print(f"[resume] loaded checkpoint from {resume_from} at step {start_step}")

    model.train()
    train_start = time.time()
    try:
        for step in range(start_step, max_iters):
            # LR schedule uses zero-based iteration index `step`.
            lr = get_lr_cosine_schedule(
                t=step,
                alpha_max=max_lr,
                alpha_min=min_lr,
                T_w=warmup_iters,
                T_c=cosine_cycle_iters,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            inputs, targets = data_loading(
                x=train_tokens,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            logits = model(inputs)
            loss = cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                gradient_clipping(model.parameters(), grad_clip)
            optimizer.step()

            # Human-readable step count (1-based) for logging/checkpoint metadata.
            completed_step = step + 1
            if completed_step % log_every == 0 or completed_step == start_step + 1:
                elapsed = max(time.time() - train_start, 1e-8)
                tokens_seen = (completed_step - start_step) * batch_size * context_length
                tokens_per_sec = tokens_seen / elapsed
                train_loss = loss.item()
                print(
                    f"[train] step={completed_step} loss={train_loss:.4f} "
                    f"lr={lr:.6g} tok/s={tokens_per_sec:.1f}"
                )
                wandb.log(
                    {
                        "step": completed_step,
                        "train/loss": train_loss,
                        "train/lr": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                    },
                    step=completed_step,
                )

            if completed_step % eval_every == 0 or completed_step == max_iters:
                # Validation is sampled over several random batches for a less noisy signal.
                model.eval()
                val_losses: list[float] = []
                with torch.no_grad():
                    for _ in range(eval_batches):
                        val_inputs, val_targets = data_loading(
                            x=valid_tokens,
                            batch_size=batch_size,
                            context_length=context_length,
                            device=device,
                        )
                        val_logits = model(val_inputs)
                        val_loss = cross_entropy(val_logits, val_targets)
                        val_losses.append(val_loss.item())
                mean_val_loss = float(np.mean(val_losses))
                val_perplexity = math.exp(mean_val_loss)
                print(
                    f"[valid] step={completed_step} loss={mean_val_loss:.4f} "
                    f"ppl={val_perplexity:.2f}"
                )
                wandb.log(
                    {
                        "step": completed_step,
                        "val/loss": mean_val_loss,
                        "val/perplexity": val_perplexity,
                    },
                    step=completed_step,
                )
                model.train()

            if completed_step % save_every == 0 or completed_step == max_iters:
                # Save model + optimizer + current step so training can resume exactly.
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=completed_step,
                    out=checkpoint_path_obj,
                )
                print(f"[ckpt] saved checkpoint to {checkpoint_path_obj} at step {completed_step}")
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    # Keep argument parsing explicit in this script so students can see all tunable knobs in one place.
    parser = argparse.ArgumentParser(description="Train TransformerLM end-to-end with W&B logging.")
    parser.add_argument("--train-tokens-path", type=str, required=True)
    parser.add_argument("--valid-tokens-path", type=str, required=True)
    parser.add_argument(
        "--token-dtype",
        type=str,
        default="uint16",
        choices=["uint16", "uint32", "int32", "int64"],
        help="Used for raw binary token files (ignored for .npy files).",
    )
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--cosine-cycle-iters", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--wandb-project", type=str, default="cs336-a1")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    cli_args = parser.parse_args()
    train(**vars(cli_args))
