"""Gradient-sensitivity probe: run a uniform-rank LoRA adapter for probe_steps steps and accumulate
per-module gradient statistics on lora_B. BUILD_SPEC.md 4.3.

Why lora_B, and what the number actually means
----------------------------------------------
B is zero-initialised, so its gradient carries signal from step 0 (dL/dA depends on B and is
therefore exactly zero at init). More usefully, for a single micro-batch the chain rule gives

    dL/dB = s * G @ A.T          where  G = dL/dW  is the gradient the *frozen* weight would receive
                                 and    s = alpha / r  is the LoRA scaling factor

so the B-gradient is not merely correlated with the frozen-weight gradient -- it is a random
*sketch* of it, with A playing the role of the random projection. With A's entries drawn from
PEFT's kaiming_uniform_(a=sqrt(5)) (per-entry variance 1/(3*d_in)),

    E ||dL/dB||_F^2 = s^2 * (r / (3 * d_in)) * ||G||_F^2

and therefore

    rms := ||dL/dB||_F / sqrt(numel(B))
         = ||dL/dB||_F / sqrt(d_out * r)
         = (s / sqrt(3)) * ||G||_F / sqrt(d_in * d_out)
         = (s / sqrt(3)) * (per-parameter RMS of the frozen-weight gradient)

i.e. rms is width-normalised in *both* dimensions and is independent of the probe rank. That is why
rms, not raw_norm, is the default: raw_norm differs from rms by exactly sqrt(d_out * r) and so ranks
wide-output modules above narrow ones for reasons unrelated to importance. (CLAUDE.md is explicit
that raw_norm must never quietly become the default. It is retained as the deliberate width-confound
ablation, not as a candidate signal.)

The sketch identity only holds while A *is* the random projection. Two consequences, both handled
below:

  - lora_A is frozen for the duration of the probe (`freeze_a=True`, the default). If A is allowed
    to train, it becomes correlated with G and the estimator picks up a module-dependent upward bias
    that no amount of averaging removes.
  - gradients are unscaled before they are read. Under fp16 autocast, `.grad` holds
    (loss scale) * (true gradient), and the dynamic loss scale changes over the course of the probe,
    so an average of raw per-step norms is a scale-weighted average that over-weights whichever
    steps happened to run at the higher scale.

Coherent vs incoherent accumulation
-----------------------------------
`rms` averages per-step norms, which estimates E||G_t|| = signal + noise (a Fisher-like second-moment
quantity). `coherent` instead norms the *accumulated* gradient, ||sum_t G_t||, which estimates the
consistent descent direction with the noise averaged out. They differ whenever a module's gradient is
large but inconsistent, and E||g||^2 = ||E g||^2 + tr(Cov) makes the gap explicit. Both are recorded,
along with their ratio (`coherence`), so the choice is an analysis decision rather than a hard-coded
one.
"""
import hashlib
import json
import time
from dataclasses import dataclass
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

SIGNAL_KEYS = ("rms", "raw_norm", "fisher", "relative", "coherent")


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
    coherence: Dict[str, float]  # module_name -> ||sum_t g_t|| / sum_t ||g_t||, in (0, 1]
    module_meta: Dict[str, dict]  # module_name -> {in_features, out_features, numel, layer_idx, proj_type}
    probe_wall_seconds: float
    probe_gpu_seconds: float
    seed: int
    split_seed: int
    freeze_a: bool
    unscaled: bool


def compute_probe_id(
    model_name: str,
    task: str,
    rank: int,
    steps: int,
    seed: int,
    split_seed: int,
    freeze_a: bool = True,
) -> str:
    """Hash of everything that changes the probe's output.

    `freeze_a` participates because a frozen-A probe and a trained-A probe measure genuinely
    different quantities (see the module docstring); they must not collide in results/probe/.
    """
    payload = {
        "model_name": model_name,
        "task": task,
        "rank": rank,
        "steps": steps,
        "seed": seed,
        "split_seed": split_seed,
        "freeze_a": bool(freeze_a),
    }
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
    freeze_a: bool = True,
) -> ProbeResult:
    set_seed(seed)
    probe_id = compute_probe_id(model_name, task, rank, steps, seed, split_seed, freeze_a)
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
    model = wrap_with_lora(
        base, rank_pattern, alpha_pattern, scaling_mode="constant_ratio", target_modules=target_modules
    )
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
    lora_b_by_name, lora_a_by_name = {}, {}
    for raw_name, module in model.named_modules():
        lora_b = getattr(module, "lora_B", None)
        lora_a = getattr(module, "lora_A", None)
        if lora_b is None or "default" not in lora_b:
            continue
        name = raw_name[len(PEFT_NAME_PREFIX) :] if raw_name.startswith(PEFT_NAME_PREFIX) else raw_name
        if name in rank_pattern:
            lora_b_by_name[name] = lora_b["default"]
            if lora_a is not None and "default" in lora_a:
                lora_a_by_name[name] = lora_a["default"]
    assert set(lora_b_by_name) == set(rank_pattern), "could not locate lora_B for every target module"

    if freeze_a:
        # Keeps dL/dB = s * G @ A.T an unbiased sketch of the frozen-weight gradient for the whole
        # probe. Without this, A adapts, becomes correlated with G, and the estimator drifts.
        assert set(lora_a_by_name) == set(rank_pattern), "could not locate lora_A for every target module"
        for mod in lora_a_by_name.values():
            mod.weight.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "probe has no trainable parameters"
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(steps * warmup_ratio), num_training_steps=steps
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    accum = {key: {name: 0.0 for name in rank_pattern} for key in SIGNAL_KEYS if key != "coherent"}
    # running sum of the raw gradient tensors, for the coherent (noise-averaged) signal
    grad_sum = {name: torch.zeros_like(mod.weight, dtype=torch.float32) for name, mod in lora_b_by_name.items()}
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
            # Undo the dynamic loss scale *before* reading .grad. Without this every recorded
            # magnitude carries a time-varying multiplicative factor, and because the factor changes
            # when the scaler backs off after an overflow, the average over steps silently becomes a
            # scale-weighted average rather than a mean gradient.
            scaler.unscale_(optimizer)
        else:
            out = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            out.loss.backward()

        # Compute this step's per-module contribution before we know whether fp16 GradScaler will
        # accept it -- an overflow step (routine in early fp16 training; that's the whole point of
        # dynamic loss scaling) can leave .grad full of inf/nan, and folding even one such step into
        # the running signal average poisons it with NaN for the rest of the run. Only merge into
        # accum once we know the scaler actually applied the step.
        step_contrib = {key: {} for key in SIGNAL_KEYS if key != "coherent"}
        step_grads = {}
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
            step_grads[name] = grad

        if use_amp:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            stepped = scaler.get_scale() >= scale_before
        else:
            optimizer.step()
            stepped = True

        if stepped:
            for key, contrib in step_contrib.items():
                for name, val in contrib.items():
                    accum[key][name] += val
            for name, g in step_grads.items():
                grad_sum[name] += g
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

    signals = {
        key: {name: accum[key][name] / valid_steps for name in rank_pattern}
        for key in accum
    }
    # coherent: norm of the mean gradient, normalised identically to rms so the two are directly
    # comparable on the same axis.
    signals["coherent"] = {
        name: float(torch.linalg.vector_norm(grad_sum[name] / valid_steps).item())
        / (grad_sum[name].numel() ** 0.5)
        for name in rank_pattern
    }
    # coherence in (0, 1]: 1.0 means every step pointed the same way, ~0 means the module's gradient
    # is dominated by batch noise and extra rank there buys nothing.
    coherence = {
        name: (signals["coherent"][name] / signals["rms"][name]) if signals["rms"][name] > 0 else 0.0
        for name in rank_pattern
    }

    result = ProbeResult(
        probe_id=probe_id,
        model_name=model_name,
        task=task,
        rank=rank,
        steps=steps,
        valid_steps=valid_steps,
        signals=signals,
        coherence=coherence,
        module_meta=module_meta,
        probe_wall_seconds=probe_wall_seconds,
        probe_gpu_seconds=probe_gpu_seconds,
        seed=seed,
        split_seed=split_seed,
        freeze_a=freeze_a,
        unscaled=True,
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
        "coherence": result.coherence,
        "module_meta": result.module_meta,
        "probe_wall_seconds": result.probe_wall_seconds,
        "probe_gpu_seconds": result.probe_gpu_seconds,
        "seed": result.seed,
        "split_seed": result.split_seed,
        "freeze_a": result.freeze_a,
        "unscaled": result.unscaled,
    }
