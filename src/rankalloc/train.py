"""Fixed-step LoRA training loop, hand-written (no Trainer) so timing and logging are unambiguous.
BUILD_SPEC.md §4.6.
"""
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch
from transformers import get_cosine_schedule_with_warmup

from rankalloc.data import TokenizedExample, collate_batch


def _clip(trainable, clip_groups, clip_mode, max_grad_norm):
    """Apply gradient clipping and return the *global* pre-clip norm for logging.

    The returned value is always the global norm regardless of mode, so step_log.jsonl stays
    comparable across clip modes and can be used to measure how often the clip actually binds.
    """
    grads = [p.grad for p in trainable if p.grad is not None]
    if not grads:
        return 0.0
    total = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in grads])
    )
    if clip_mode == "global":
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    elif clip_mode == "per_module":
        for group in clip_groups:
            torch.nn.utils.clip_grad_norm_(group, max_grad_norm)
    return float(total)


@dataclass
class TrainResult:
    status: str  # "ok" | "oom"
    final_step: int
    train_wall_seconds: float
    train_gpu_seconds: float
    tokens_seen: int
    peak_vram_mb: float
    held_out_loss_curve: List[dict] = field(default_factory=list)


def train(
    model,
    train_examples: List[TokenizedExample],
    pad_token_id: int,
    *,
    max_steps: int,
    lr: float,
    warmup_ratio: float,
    max_grad_norm: float,
    micro_batch: int,
    device: str,
    run_dir: Path,
    eval_every: int = 50,
    eval_fn: Optional[Callable] = None,
) -> TrainResult:
    """eval_fn(model) -> dict is called every eval_every steps (and on the final step) for the
    periodic held-out-loss learning curve -- these points are a result (§4.6), not a diagnostic.
    """
    use_amp = str(device).startswith("cuda")
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(max_steps * warmup_ratio), num_training_steps=max_steps
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    if data_order_seed is not None:
        train_examples = list(train_examples)
        random.Random(data_order_seed).shuffle(train_examples)

    n_examples = len(train_examples)
    assert n_examples > 0, "no training examples available"

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokens_seen = 0
    held_out_loss_curve = []
    final_step = 0
    status = "ok"

    wall_start = time.perf_counter()
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        gpu_start_evt = torch.cuda.Event(enable_timing=True)
        gpu_start_evt.record()

    try:
        with (run_dir / "step_log.jsonl").open("w") as step_log:
            for step in range(max_steps):
                batch = [train_examples[(step * micro_batch + i) % n_examples] for i in range(micro_batch)]
                input_ids, labels, attention_mask = collate_batch(batch, pad_token_id, device)
                tokens_seen += int(attention_mask.sum().item())

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                    scaler.scale(out.loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = _clip(trainable, clip_groups, clip_mode, max_grad_norm)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    # GradScaler silently skips optimizer.step() on an inf/nan-gradient step, signalled
                    # by a scale decrease -- only step the LR schedule when the optimizer actually
                    # stepped, or the schedule drifts out of sync with real progress (unlike probe.py's
                    # diagnostic-only loop, this matters here: the schedule shape is a real result).
                    stepped = scaler.get_scale() >= scale_before
                else:
                    out = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                    out.loss.backward()
                    grad_norm = _clip(trainable, clip_groups, clip_mode, max_grad_norm)
                    optimizer.step()
                    stepped = True
                if stepped:
                    scheduler.step()

                if use_amp:
                    step_evt = torch.cuda.Event(enable_timing=True)
                    step_evt.record()
                    torch.cuda.synchronize()
                    cumulative_gpu_seconds = gpu_start_evt.elapsed_time(step_evt) / 1000.0
                else:
                    cumulative_gpu_seconds = 0.0

                step_log.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(out.loss.item()),
                            "lr": scheduler.get_last_lr()[0],
                            "grad_norm": float(grad_norm),
                            "tokens_seen": tokens_seen,
                            "cumulative_gpu_seconds": cumulative_gpu_seconds,
                        }
                    )
                    + "\n"
                )

                final_step = step + 1
                if eval_fn is not None and (final_step % eval_every == 0 or final_step == max_steps):
                    held_out = eval_fn(model)
                    held_out_loss_curve.append({"step": final_step, **held_out})
                    model.train()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            status = "oom"
        else:
            raise

    if use_amp:
        end_evt = torch.cuda.Event(enable_timing=True)
        end_evt.record()
        torch.cuda.synchronize()
        train_gpu_seconds = gpu_start_evt.elapsed_time(end_evt) / 1000.0
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        train_gpu_seconds = 0.0
        peak_vram_mb = 0.0
    train_wall_seconds = time.perf_counter() - wall_start

    return TrainResult(
        status=status,
        final_step=final_step,
        train_wall_seconds=train_wall_seconds,
        train_gpu_seconds=train_gpu_seconds,
        tokens_seen=tokens_seen,
        peak_vram_mb=peak_vram_mb,
        held_out_loss_curve=held_out_loss_curve,
    )
