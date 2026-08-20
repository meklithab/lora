"""Gradient-sensitivity probe: train a uniform-rank LoRA adapter for probe_steps steps and accumulate
per-module gradient statistics on lora_B. BUILD_SPEC.md §4.3.

B is zero-initialised, so its gradient carries signal from step 0 (A's gradient is zero at init since
dL/dA depends on B). rms is the default probe signal -- CLAUDE.md is explicit that raw_norm must
never quietly become the default, since it scales with module width and would make wide MLP
projections dominate for reasons unrelated to importance.
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from rankalloc.config import DataConfig
from rankalloc.data import collate_batch, load_task
from rankalloc.modeling import (
    DEFAULT_TARGET_MODULES,
    PEFT_NAME_PREFIX,
    discover_module_specs,
    load_base_model,
    wrap_with_lora,
)
from rankalloc.seeding import set_seed

SIGNAL_KEYS = ("rms", "raw_norm", "fisher", "relative")


@dataclass
class ProbeResult:
    probe_id: str
    model_name: str
    task: str
    rank: int
    steps: int
    valid_steps: int  # steps actually folded into signals -- may be < steps if fp16 GradScaler
    # skipped an early overflow step (routine, not an error); see run_probe's docstring comment
    signals: Dict[str, Dict[str, float]]  # signal_name -> {module_name: value}
    module_meta: Dict[str, dict]  # module_name -> {in_features, out_features, numel, layer_idx, proj_type}
    probe_wall_seconds: float
    probe_gpu_seconds: float
    seed: int
    split_seed: int


def compute_probe_id(model_name: str, task: str, rank: int, steps: int, seed: int, split_seed: int) -> str:
    payload = {"model_name": model_name, "task": task, "rank": rank, "steps": steps, "seed": seed, "split_seed": split_seed}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:12]


def _proj_type(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def run_probe(
    model_name: str,
    task: str,
    rank: int,
    steps: int,
    lr: float,
    warmup_ratio: float,
    seed: int,
    split_seed: int,
    max_seq_len: int = 512,
    micro_batch: int = 4,
    alpha_ratio: int = 2,
    target_modules=DEFAULT_TARGET_MODULES,
    device: Optional[str] = "cuda",
    dtype=torch.float16,
    out_dir: Optional[Path] = None,
) -> ProbeResult:
    set_seed(seed)
    probe_id = compute_probe_id(model_name, task, rank, steps, seed, split_seed)
    use_amp = device is not None and str(device).startswith("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_cfg = DataConfig(task=task, max_seq_len=max_seq_len, split_seed=split_seed)
    bundle = load_task(data_cfg, tokenizer)

    base = load_base_model(model_name, dtype=dtype, device=device, gradient_checkpointing=use_amp)
    specs = discover_module_specs(base, target_modules)
    rank_pattern = {s.name: rank for s in specs}
    alpha_pattern = {s.name: float(alpha_ratio * rank) for s in specs}
    model = wrap_with_lora(base, rank_pattern, alpha_pattern, scaling_mode="constant_ratio", target_modules=target_modules)
    model.train()

    module_meta = {
        s.name: {
            "in_features": s.d_in,
            "out_features": s.d_out,
            "numel": s.d_in * s.d_out,
            "layer_idx": s.layer_idx,
            "proj_type": _proj_type(s.name),
        }
        for s in specs
    }
    # lora_B lives on the same named module as lora_A, keyed by the base module's own (pre-wrap)
    # name -- get_peft_model() prefixes every live name with "base_model.model." (see
    # modeling.verify_live_model's docstring), so strip that back off before matching rank_pattern.
    lora_b_by_name = {}
    for raw_name, module in model.named_modules():
        lora_b = getattr(module, "lora_B", None)
        if lora_b is None or "default" not in lora_b:
            continue
        name = raw_name[len(PEFT_NAME_PREFIX) :] if raw_name.startswith(PEFT_NAME_PREFIX) else raw_name
        if name in rank_pattern:
            lora_b_by_name[name] = lora_b["default"]
    assert set(lora_b_by_name) == set(rank_pattern), "could not locate lora_B for every target module"

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(steps * warmup_ratio), num_training_steps=steps
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    accum = {key: {name: 0.0 for name in rank_pattern} for key in SIGNAL_KEYS}
    eps = 1e-8

    examples = bundle.train
    n_examples = len(examples)
    assert n_examples > 0, f"no training examples available for task={task!r}"

    wall_start = time.perf_counter()
    if use_amp:
        torch.cuda.synchronize()
        gpu_start_evt = torch.cuda.Event(enable_timing=True)
        gpu_end_evt = torch.cuda.Event(enable_timing=True)
        gpu_start_evt.record()

    valid_steps = 0
    for step in range(steps):
        batch_examples = [examples[(step * micro_batch + i) % n_examples] for i in range(micro_batch)]
        input_ids, labels, attention_mask = collate_batch(batch_examples, tokenizer.pad_token_id, device)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            scaler.scale(out.loss).backward()
        else:
            out = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            out.loss.backward()

        # Compute this step's per-module contribution before we know whether fp16 GradScaler will
        # accept it -- an overflow step (routine in early fp16 training; that's the whole point of
        # dynamic loss scaling) can leave .grad full of inf/nan, and folding even one such step into
        # the running signal average poisons it with NaN for the rest of the run. Only merge into
        # accum once we know the scaler actually applied the step.
        step_contrib = {key: {} for key in SIGNAL_KEYS}
        for name, b_param in lora_b_by_name.items():
            grad = b_param.weight.grad
            if grad is None:
                continue
            grad = grad.detach().float()
            numel = grad.numel()
            g_norm = torch.linalg.vector_norm(grad).item()
            w_norm = torch.linalg.vector_norm(b_param.weight.detach().float()).item()
            step_contrib["rms"][name] = g_norm / (numel**0.5)
            step_contrib["raw_norm"][name] = g_norm
            step_contrib["fisher"][name] = float((grad**2).sum().item()) / numel
            step_contrib["relative"][name] = g_norm / (w_norm + eps)

        if use_amp:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            stepped = scaler.get_scale() >= scale_before
        else:
            optimizer.step()
            stepped = True

        if stepped:
            for key in SIGNAL_KEYS:
                for name, val in step_contrib[key].items():
                    accum[key][name] += val
            valid_steps += 1
            scheduler.step()

    assert valid_steps > 0, (
        "every probe step was skipped by the fp16 GradScaler (all-inf/nan gradients) -- this points "
        "at something wrong with lr/init, not routine early-step overflow"
    )

    if use_amp:
        gpu_end_evt.record()
        torch.cuda.synchronize()
        probe_gpu_seconds = gpu_start_evt.elapsed_time(gpu_end_evt) / 1000.0
    else:
        probe_gpu_seconds = 0.0
    probe_wall_seconds = time.perf_counter() - wall_start

    signals = {key: {name: accum[key][name] / valid_steps for name in rank_pattern} for key in SIGNAL_KEYS}

    result = ProbeResult(
        probe_id=probe_id,
        model_name=model_name,
        task=task,
        rank=rank,
        steps=steps,
        valid_steps=valid_steps,
        signals=signals,
        module_meta=module_meta,
        probe_wall_seconds=probe_wall_seconds,
        probe_gpu_seconds=probe_gpu_seconds,
        seed=seed,
        split_seed=split_seed,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{probe_id}.json").write_text(json.dumps(field_asdict(result), indent=2, sort_keys=True))

    return result


def field_asdict(result: ProbeResult) -> dict:
    return {
        "probe_id": result.probe_id,
        "model_name": result.model_name,
        "task": result.task,
        "rank": result.rank,
        "steps": result.steps,
        "valid_steps": result.valid_steps,
        "signals": result.signals,
        "module_meta": result.module_meta,
        "probe_wall_seconds": result.probe_wall_seconds,
        "probe_gpu_seconds": result.probe_gpu_seconds,
        "seed": result.seed,
        "split_seed": result.split_seed,
    }
