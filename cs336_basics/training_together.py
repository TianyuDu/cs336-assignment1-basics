import argparse
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


SAFE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "TOKENIZERS_PARALLELISM",
    "WANDB_DIR",
    "WANDB_ENTITY",
    "WANDB_MODE",
    "WANDB_PROJECT",
)
MAX_GIT_STATUS_LINES = 200


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _run_command(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    stdout = result.stdout.strip()
    return stdout or None


def _summarize_git_status(status_lines: list[str]) -> dict[str, int]:
    summary = {
        "total_entries": len(status_lines),
        "untracked_entries": 0,
        "staged_entries": 0,
        "unstaged_entries": 0,
        "modified_entries": 0,
        "deleted_entries": 0,
        "renamed_entries": 0,
    }
    for line in status_lines:
        if line.startswith("??"):
            summary["untracked_entries"] += 1
            continue
        index_status = line[0]
        worktree_status = line[1]
        if index_status != " ":
            summary["staged_entries"] += 1
        if worktree_status != " ":
            summary["unstaged_entries"] += 1
        if "M" in (index_status, worktree_status):
            summary["modified_entries"] += 1
        if "D" in (index_status, worktree_status):
            summary["deleted_entries"] += 1
        if "R" in (index_status, worktree_status):
            summary["renamed_entries"] += 1
    return summary


def _collect_git_metadata(repo_root: Path) -> dict[str, Any]:
    if _run_command(["git", "rev-parse", "--is-inside-work-tree"], repo_root) != "true":
        return {
            "available": False,
            "repo_root": str(repo_root.resolve()),
        }

    status_output = _run_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        repo_root,
    )
    status_lines = status_output.splitlines() if status_output else []
    stored_status_lines = status_lines[:MAX_GIT_STATUS_LINES]
    return {
        "available": True,
        "repo_root": str(repo_root.resolve()),
        "branch": _run_command(["git", "branch", "--show-current"], repo_root),
        "commit": _run_command(["git", "rev-parse", "HEAD"], repo_root),
        "commit_short": _run_command(["git", "rev-parse", "--short", "HEAD"], repo_root),
        "describe": _run_command(["git", "describe", "--always", "--dirty", "--tags"], repo_root),
        "last_commit_subject": _run_command(["git", "log", "-1", "--format=%s"], repo_root),
        "remote_origin_url": _run_command(["git", "remote", "get-url", "origin"], repo_root),
        "dirty": bool(status_lines),
        "status_summary": _summarize_git_status(status_lines),
        "status_lines": stored_status_lines,
        "status_lines_truncated": len(stored_status_lines) < len(status_lines),
        "unstaged_diff_shortstat": _run_command(["git", "diff", "--shortstat"], repo_root),
        "staged_diff_shortstat": _run_command(["git", "diff", "--cached", "--shortstat"], repo_root),
    }


def _collect_safe_env_metadata() -> dict[str, str]:
    return {key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ}


def _get_selected_cuda_index(device: str) -> int:
    device_obj = torch.device(device)
    if device_obj.index is not None:
        return device_obj.index
    return torch.cuda.current_device()


def _collect_runtime_metadata(*, device: str, dtype: str, seed: int) -> dict[str, Any]:
    runtime_metadata: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "working_directory": str(Path.cwd().resolve()),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "seed": seed,
        "device_requested": device,
        "dtype_requested": dtype,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "wandb_version": wandb.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_initial_seed": torch.initial_seed(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_available": torch.backends.cudnn.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "environment": _collect_safe_env_metadata(),
        "command_line": " ".join(sys.argv),
    }
    if hasattr(torch, "get_float32_matmul_precision"):
        runtime_metadata["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    try:
        runtime_metadata["cuda_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    except AttributeError:
        runtime_metadata["cuda_allow_tf32"] = None
    try:
        runtime_metadata["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    except AttributeError:
        runtime_metadata["cudnn_allow_tf32"] = None
    if torch.cuda.is_available():
        cuda_devices: list[dict[str, Any]] = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            cuda_devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_mb": round(props.total_memory / (1024**2), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                }
            )
        runtime_metadata["cuda_device_count"] = len(cuda_devices)
        runtime_metadata["cuda_devices"] = cuda_devices
        if device.startswith("cuda"):
            selected_idx = _get_selected_cuda_index(device)
            runtime_metadata["selected_cuda_device_index"] = selected_idx
            runtime_metadata["selected_cuda_device_name"] = cuda_devices[selected_idx]["name"]
    return runtime_metadata


def _collect_dataset_metadata(path: Path, tokens: np.ndarray) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "storage_format": "npy" if path.suffix == ".npy" else "raw_binary",
        "file_size_bytes": path.stat().st_size if path.exists() else None,
        "token_count": int(tokens.shape[0]),
        "dtype": str(tokens.dtype),
        "ndim": int(tokens.ndim),
    }


def _collect_model_metadata(
    model: torch.nn.Module,
    model_parameters: list[torch.nn.Parameter],
) -> dict[str, Any]:
    total_parameters = sum(parameter.numel() for parameter in model_parameters)
    trainable_parameters = sum(parameter.numel() for parameter in model_parameters if parameter.requires_grad)
    parameter_breakdown: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        root_module = name.split(".", 1)[0]
        parameter_breakdown[root_module] = parameter_breakdown.get(root_module, 0) + parameter.numel()

    first_parameter = model_parameters[0] if model_parameters else None
    return {
        "class_name": model.__class__.__name__,
        "parameter_count_total": int(total_parameters),
        "parameter_count_total_millions": round(total_parameters / 1_000_000, 4),
        "parameter_count_trainable": int(trainable_parameters),
        "parameter_count_non_trainable": int(total_parameters - trainable_parameters),
        "parameter_breakdown_by_root_module": parameter_breakdown,
        "parameter_dtype": str(first_parameter.dtype) if first_parameter is not None else None,
        "parameter_device": str(first_parameter.device) if first_parameter is not None else None,
    }


def _compute_grad_norm(model_parameters: list[torch.nn.Parameter]) -> float | None:
    total_norm_sq = 0.0
    saw_gradient = False
    for parameter in model_parameters:
        if parameter.grad is None:
            continue
        total_norm_sq += float(parameter.grad.detach().float().pow(2).sum().item())
        saw_gradient = True
    if not saw_gradient:
        return None
    return math.sqrt(total_norm_sq)


def _collect_cuda_memory_metrics(device_obj: torch.device) -> dict[str, float]:
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        return {}

    device_index = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    return {
        "system/cuda_memory_allocated_mb": round(torch.cuda.memory_allocated(device_index) / (1024**2), 2),
        "system/cuda_memory_reserved_mb": round(torch.cuda.memory_reserved(device_index) / (1024**2), 2),
        "system/cuda_max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device_index) / (1024**2), 2),
        "system/cuda_max_memory_reserved_mb": round(torch.cuda.max_memory_reserved(device_index) / (1024**2), 2),
        "system/cuda_memory_free_mb": round(free_bytes / (1024**2), 2),
        "system/cuda_memory_total_mb": round(total_bytes / (1024**2), 2),
    }


def _configure_wandb_mode(wandb_mode: str | None) -> None:
    if wandb_mode is None:
        return
    if wandb_mode in {"offline", "disabled"}:
        os.environ["WANDB_MODE"] = wandb_mode
        return
    if wandb_mode == "online":
        os.environ.pop("WANDB_MODE", None)
        return
    raise ValueError("--wandb-mode must be one of: online, offline, disabled.")


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
    fixed_batch: bool = False,
    target_val_loss: float | None = None,
    wandb_project: str = "cs336-a1",
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    wandb_job_type: str | None = None,
    wandb_notes: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_mode: str | None = None,
) -> dict[str, Any]:
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
    if device.startswith("mps") and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("Requested MPS device but MPS is not available.")

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
    repo_root = Path(__file__).resolve().parents[1]
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
        "fixed_batch": fixed_batch,
        "target_val_loss": target_val_loss,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "wandb_group": wandb_group,
        "wandb_job_type": wandb_job_type,
        "wandb_notes": wandb_notes,
        "wandb_tags": wandb_tags,
        "wandb_mode": wandb_mode,
    }
    # Keep W&B setup direct and visible for students: one init, periodic logs, one finish.
    _configure_wandb_mode(wandb_mode)
    run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        group=wandb_group,
        job_type=wandb_job_type,
        notes=wandb_notes,
        tags=wandb_tags,
        config=run_config,
    )
    run.define_metric("trainer/global_step")
    run.define_metric("trainer/elapsed_wall_time_sec")
    for metric_pattern in ("train/*", "val/*", "system/*", "checkpoint/*"):
        run.define_metric(metric_pattern, step_metric="trainer/global_step")
    run.summary["run/status"] = "running"
    run.summary["wandb/run_id"] = run.id
    run.summary["wandb/run_name"] = run.name
    if getattr(run, "url", None):
        run.summary["wandb/run_url"] = run.url

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
    model_parameters = list(model.parameters())
    device_obj = torch.device(device)

    optimizer = AdamW(
        model_parameters,
        lr=max_lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )

    start_step = 0
    if resume_from:
        start_step = load_checkpoint(resume_from, model, optimizer)
        print(f"[resume] loaded checkpoint from {resume_from} at step {start_step}")

    batch_tokens = batch_size * context_length
    derived_metadata = {
        "resume_start_step": start_step,
        "tokens_per_train_step": batch_tokens,
        "tokens_per_validation": eval_batches * batch_tokens,
        "approx_train_steps_per_dataset_pass": len(train_tokens) / batch_tokens,
        "approx_valid_steps_per_dataset_pass": len(valid_tokens) / batch_tokens,
    }
    extra_run_metadata = {
        "runtime_metadata": _collect_runtime_metadata(device=device, dtype=dtype, seed=seed),
        "git_metadata": _collect_git_metadata(repo_root),
        "dataset_metadata": {
            "train": _collect_dataset_metadata(train_path, train_tokens),
            "valid": _collect_dataset_metadata(valid_path, valid_tokens),
        },
        "model_metadata": _collect_model_metadata(model, model_parameters),
        "derived_metadata": derived_metadata,
        "checkpoint_metadata": {
            "checkpoint_path": str(checkpoint_path_obj.resolve()),
            "resume_from": str(Path(resume_from).resolve()) if resume_from is not None else None,
        },
        "wandb_metadata": {
            "run_id": run.id,
            "run_name": run.name,
            "run_url": getattr(run, "url", None),
        },
    }
    run.config.update(_to_serializable(extra_run_metadata), allow_val_change=True)
    if device_obj.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_obj)

    model.train()
    fixed_inputs: torch.Tensor | None = None
    fixed_targets: torch.Tensor | None = None
    if fixed_batch:
        fixed_inputs, fixed_targets = data_loading(
            x=train_tokens,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
    train_start = time.perf_counter()
    best_val_loss = math.inf
    best_val_step: int | None = None
    last_train_loss: float | None = None
    last_train_accuracy: float | None = None
    last_val_loss: float | None = None
    last_completed_step = start_step
    run_status = "completed"
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

            completed_step = step + 1
            should_log_train = (
                completed_step % log_every == 0
                or completed_step == start_step + 1
                or completed_step == max_iters
            )
            should_eval = completed_step % eval_every == 0 or completed_step == max_iters
            should_save = completed_step % save_every == 0 or completed_step == max_iters
            stop_early = False

            step_start = time.perf_counter()
            if fixed_inputs is not None and fixed_targets is not None:
                inputs, targets = fixed_inputs, fixed_targets
                data_loading_time = 0.0
            else:
                inputs, targets = data_loading(
                    x=train_tokens,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device,
                )
                data_loading_time = time.perf_counter() - step_start

            forward_start = time.perf_counter()
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            forward_loss_time = time.perf_counter() - forward_start

            optimizer.zero_grad()
            backward_start = time.perf_counter()
            loss.backward()
            grad_norm_pre_clip = _compute_grad_norm(model_parameters) if should_log_train else None
            if grad_clip > 0:
                gradient_clipping(model_parameters, grad_clip)
            grad_norm_post_clip = (
                _compute_grad_norm(model_parameters) if should_log_train and grad_clip > 0 else None
            )
            optimizer.step()
            backward_optimizer_time = time.perf_counter() - backward_start
            step_time = time.perf_counter() - step_start
            train_loss = float(loss.item())
            last_train_loss = train_loss

            # Human-readable step count (1-based) for logging/checkpoint metadata.
            last_completed_step = completed_step
            if should_log_train:
                elapsed = max(time.perf_counter() - train_start, 1e-8)
                tokens_seen_since_resume = (completed_step - start_step) * batch_tokens
                total_tokens_seen = completed_step * batch_tokens
                tokens_per_sec = tokens_seen_since_resume / elapsed
                train_perplexity = math.exp(min(train_loss, 20.0))
                train_accuracy = float((logits.argmax(dim=-1) == targets).float().mean().item())
                last_train_accuracy = train_accuracy
                print(
                    f"[train] step={completed_step} loss={train_loss:.4f} "
                    f"lr={lr:.6g} tok/s={tokens_per_sec:.1f} step_time={step_time:.3f}s"
                )
                train_metrics: dict[str, Any] = {
                    "step": completed_step,
                    "trainer/global_step": completed_step,
                    "trainer/elapsed_wall_time_sec": elapsed,
                    "train/elapsed_wall_time_sec": elapsed,
                    "train/loss": train_loss,
                    "train/perplexity": train_perplexity,
                    "train/accuracy": train_accuracy,
                    "train/lr": lr,
                    "train/batch_tokens": batch_tokens,
                    "train/tokens_seen_since_resume": tokens_seen_since_resume,
                    "train/total_tokens_seen": total_tokens_seen,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/approx_dataset_passes": total_tokens_seen / len(train_tokens),
                    "train/step_time_sec": step_time,
                    "train/data_loading_time_sec": data_loading_time,
                    "train/forward_loss_time_sec": forward_loss_time,
                    "train/backward_optimizer_time_sec": backward_optimizer_time,
                }
                if grad_norm_pre_clip is not None:
                    train_metrics["train/grad_norm_pre_clip"] = grad_norm_pre_clip
                if grad_norm_post_clip is not None:
                    train_metrics["train/grad_norm_post_clip"] = grad_norm_post_clip
                    train_metrics["train/grad_was_clipped"] = bool(grad_norm_pre_clip > grad_clip)
                train_metrics.update(_collect_cuda_memory_metrics(device_obj))
                run.log(train_metrics, step=completed_step)

            if should_eval:
                # Validation is sampled over several random batches for a less noisy signal.
                model.eval()
                val_losses: list[float] = []
                eval_start = time.perf_counter()
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
                eval_time = time.perf_counter() - eval_start
                mean_val_loss = float(np.mean(val_losses))
                val_perplexity = math.exp(mean_val_loss)
                last_val_loss = mean_val_loss
                elapsed = max(time.perf_counter() - train_start, 1e-8)
                print(
                    f"[valid] step={completed_step} loss={mean_val_loss:.4f} "
                    f"ppl={val_perplexity:.2f} eval_time={eval_time:.3f}s"
                )
                val_metrics: dict[str, Any] = {
                    "step": completed_step,
                    "trainer/global_step": completed_step,
                    "trainer/elapsed_wall_time_sec": elapsed,
                    "val/elapsed_wall_time_sec": elapsed,
                    "val/loss": mean_val_loss,
                    "val/perplexity": val_perplexity,
                    "val/eval_time_sec": eval_time,
                    "val/tokens_evaluated": eval_batches * batch_tokens,
                    "val/approx_dataset_passes": (eval_batches * batch_tokens) / len(valid_tokens),
                }
                val_metrics.update(_collect_cuda_memory_metrics(device_obj))
                run.log(val_metrics, step=completed_step)
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_val_step = completed_step
                    run.summary["val/best_loss"] = best_val_loss
                    run.summary["val/best_step"] = best_val_step
                    run.summary["val/best_perplexity"] = val_perplexity
                if target_val_loss is not None and mean_val_loss <= target_val_loss:
                    print(
                        f"[valid] reached target val loss {target_val_loss:.4f} "
                        f"at step {completed_step}; stopping early"
                    )
                    run.summary["val/target_loss"] = target_val_loss
                    stop_early = True
                model.train()

            if should_save or stop_early:
                # Save model + optimizer + current step so training can resume exactly.
                save_start = time.perf_counter()
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=completed_step,
                    out=checkpoint_path_obj,
                )
                save_time = time.perf_counter() - save_start
                checkpoint_size_bytes = checkpoint_path_obj.stat().st_size if checkpoint_path_obj.exists() else None
                elapsed = max(time.perf_counter() - train_start, 1e-8)
                checkpoint_metrics: dict[str, Any] = {
                    "trainer/global_step": completed_step,
                    "trainer/elapsed_wall_time_sec": elapsed,
                    "checkpoint/last_saved_step": completed_step,
                    "checkpoint/save_time_sec": save_time,
                }
                if checkpoint_size_bytes is not None:
                    checkpoint_metrics["checkpoint/file_size_mb"] = checkpoint_size_bytes / (1024**2)
                    run.summary["checkpoint/last_saved_size_bytes"] = checkpoint_size_bytes
                run.log(checkpoint_metrics, step=completed_step)
                run.summary["checkpoint/last_saved_step"] = completed_step
                run.summary["checkpoint/last_saved_path"] = str(checkpoint_path_obj.resolve())
                print(f"[ckpt] saved checkpoint to {checkpoint_path_obj} at step {completed_step}")
            if stop_early:
                break
    except BaseException as exc:
        run_status = "failed"
        run.summary["run/exception_type"] = type(exc).__name__
        run.summary["run/exception_message"] = str(exc)
        raise
    finally:
        total_elapsed = max(time.perf_counter() - train_start, 0.0)
        total_tokens_seen = last_completed_step * batch_tokens
        run.summary["run/status"] = run_status
        run.summary["trainer/resumed"] = resume_from is not None
        run.summary["trainer/start_step"] = start_step
        run.summary["trainer/final_step"] = last_completed_step
        run.summary["trainer/total_wall_time_sec"] = total_elapsed
        run.summary["trainer/final_tokens_seen"] = total_tokens_seen
        run.summary["trainer/final_approx_dataset_passes"] = total_tokens_seen / len(train_tokens)
        if last_train_loss is not None:
            run.summary["train/final_loss"] = last_train_loss
            run.summary["train/final_perplexity"] = math.exp(min(last_train_loss, 20.0))
        if last_train_accuracy is not None:
            run.summary["train/final_accuracy"] = last_train_accuracy
        if last_val_loss is not None:
            run.summary["val/final_loss"] = last_val_loss
            run.summary["val/final_perplexity"] = math.exp(last_val_loss)
        if best_val_step is not None:
            run.summary["val/best_step"] = best_val_step
            run.summary["val/best_loss"] = best_val_loss
            run.summary["val/best_perplexity"] = math.exp(best_val_loss)
        run.finish()

    summary: dict[str, Any] = {
        "status": run_status,
        "final_step": last_completed_step,
        "checkpoint_path": str(checkpoint_path_obj.resolve()),
        "final_train_loss": last_train_loss,
        "final_train_accuracy": last_train_accuracy,
        "final_val_loss": last_val_loss,
        "best_val_loss": best_val_loss if best_val_step is not None else None,
        "best_val_step": best_val_step,
    }
    if last_train_loss is not None:
        summary["final_train_perplexity"] = math.exp(min(last_train_loss, 20.0))
    if last_val_loss is not None:
        summary["final_val_perplexity"] = math.exp(last_val_loss)
    if best_val_step is not None:
        summary["best_val_perplexity"] = math.exp(best_val_loss)
    return summary


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
    parser.add_argument(
        "--fixed-batch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse one sampled training batch on every step.",
    )
    parser.add_argument(
        "--target-val-loss",
        type=float,
        default=None,
        help="Stop early once validation loss reaches this threshold.",
    )
    parser.add_argument("--wandb-project", type=str, default="cs336-a1")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-job-type", type=str, default=None)
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="Optional W&B tags to help group related experiments.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional WANDB_MODE override for this run.",
    )
    cli_args = parser.parse_args()
    train(**vars(cli_args))
