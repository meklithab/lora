"""CLI: run one full training+eval condition from a config. BUILD_SPEC.md §4.6/§4.7/§5.

--smoke: 20 steps, 8 held-out/generation examples, must finish under 3 minutes and touch every code
path (allocation -> model build -> live verification -> training -> held-out loss -> generation eval
-> results.csv row).
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import AutoTokenizer

from rankalloc.allocation import budget_for_uniform_rank, solve_allocation, strategy_weights
from rankalloc.config import RunConfig, apply_overrides, from_yaml
from rankalloc.config import run_id as compute_run_id
from rankalloc.data import load_task
from rankalloc.evaluate import compute_gsm8k_generation, compute_held_out_loss
from rankalloc.io_utils import atomic_append_csv, run_dir as get_run_dir
from rankalloc.modeling import alloc_json_payload, build_model, discover_module_specs, load_base_model, verify_live_model
from rankalloc.seeding import set_seed
from rankalloc.train import train as train_loop

RESULT_FIELDS = [
    "run_id", "condition", "seed", "strategy", "signal", "temperature", "scaling_mode",
    "budget_rank", "adapter_params_verified", "budget_abs_error", "budget_rel_error",
    "train_tokens", "supervised_tokens", "max_steps", "loss_token_weighted", "loss_example_mean",
    "gsm8k_strict", "gsm8k_flexible", "train_gpu_seconds", "train_wall_seconds", "samples_per_sec",
    "eval_gpu_seconds", "probe_gpu_seconds", "peak_vram_mb", "gpu_name", "status", "git_sha", "timestamp",
]


def load_probe_signal(probe_id: str, signal_key: str):
    data = json.loads((Path("results/probe") / f"{probe_id}.json").read_text())
    return data["signals"][signal_key], data.get("probe_gpu_seconds", 0.0)


def build_allocation(specs, cfg):
    budget = budget_for_uniform_rank(specs, cfg.alloc.budget_rank)
    signal, probe_gpu_seconds = None, 0.0
    if cfg.alloc.strategy in ("gradnorm_prop", "gradnorm_inverse"):
        assert cfg.alloc.probe_id, f"strategy={cfg.alloc.strategy!r} requires alloc.probe_id"
        signal, probe_gpu_seconds = load_probe_signal(cfg.alloc.probe_id, cfg.alloc.signal)
    weights = strategy_weights(cfg.alloc.strategy, specs, signal=signal, seed=cfg.seed, lambda_decay=cfg.alloc.lambda_decay)
    alloc = solve_allocation(specs, weights, budget, r_min=cfg.alloc.r_min, r_max=cfg.alloc.r_max, temperature=cfg.alloc.temperature)
    alloc.strategy = cfg.alloc.strategy
    alloc.signal = cfg.alloc.signal if signal is not None else None
    alloc.probe_id = cfg.alloc.probe_id
    return alloc, probe_gpu_seconds


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def gpu_time(device, fn):
    use_amp = str(device).startswith("cuda")
    if use_amp:
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
    result = fn()
    if use_amp:
        end_evt.record()
        torch.cuda.synchronize()
        return result, start_evt.elapsed_time(end_evt) / 1000.0
    return result, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--condition", default="run")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = from_yaml(args.config) if args.config else RunConfig()
    cfg = apply_overrides(cfg, args.overrides)
    if args.smoke:
        cfg = apply_overrides(cfg, ["optim.max_steps=20", "data.val_loss_n=8", "data.val_gen_n=8"])

    rid = compute_run_id(cfg)
    out_dir = get_run_dir(rid)
    device = args.device
    use_amp = device.startswith("cuda")
    dtype = torch.float16 if use_amp else torch.float32
    gpu_name = torch.cuda.get_device_name(0) if use_amp else "cpu"
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bundle = load_task(cfg.data, tokenizer)

    status = "ok"
    metrics = {
        "run_id": rid, "condition": args.condition, "seed": cfg.seed, "strategy": cfg.alloc.strategy,
        "signal": None, "temperature": cfg.alloc.temperature, "scaling_mode": cfg.scaling.mode,
        "budget_rank": cfg.alloc.budget_rank, "adapter_params_verified": False, "budget_abs_error": 0.0,
        "budget_rel_error": 0.0, "train_tokens": bundle.stats.get("total_tokens", 0),
        "supervised_tokens": bundle.stats.get("supervised_tokens", 0), "max_steps": cfg.optim.max_steps,
        "loss_token_weighted": None, "loss_example_mean": None, "gsm8k_strict": None, "gsm8k_flexible": None,
        "train_gpu_seconds": 0.0, "train_wall_seconds": 0.0, "samples_per_sec": 0.0, "eval_gpu_seconds": 0.0,
        "probe_gpu_seconds": 0.0, "peak_vram_mb": 0.0, "gpu_name": gpu_name, "status": status,
        "git_sha": git_sha(), "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if cfg.alloc.strategy == "zero_shot":
        model = load_base_model(cfg.model_name, dtype=dtype, device=device, gradient_checkpointing=False)
        model.eval()
        metrics["adapter_params_verified"] = True
    else:
        base_for_specs = load_base_model(cfg.model_name, dtype=dtype, device=None, gradient_checkpointing=False)
        specs = discover_module_specs(base_for_specs, cfg.target_modules)
        del base_for_specs

        alloc, probe_gpu_seconds = build_allocation(specs, cfg)
        model, alpha_pattern = build_model(
            cfg.model_name, alloc.rank_pattern, cfg.scaling.mode, cfg.scaling.alpha_ratio, cfg.scaling.fixed_alpha,
            target_modules=cfg.target_modules, dtype=dtype, device=device, gradient_checkpointing=use_amp,
        )
        verified = verify_live_model(model, alloc.rank_pattern, alpha_pattern)
        assert verified["adapter_params_total"] == alloc.params_total

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "alloc.json").write_text(json.dumps(alloc_json_payload(alloc, alpha_pattern, cfg.scaling.mode), indent=2, default=str))

        metrics.update(
            signal=alloc.signal, adapter_params_verified=verified["adapter_params_verified"],
            budget_abs_error=alloc.abs_error, budget_rel_error=alloc.rel_error, probe_gpu_seconds=probe_gpu_seconds,
        )

        eval_every = max(1, min(50, cfg.optim.max_steps // 4))

        def eval_fn(m):
            return compute_held_out_loss(m, bundle.val_loss, tokenizer.pad_token_id, device, micro_batch=cfg.optim.micro_batch, use_amp=use_amp)

        train_result = train_loop(
            model, bundle.train, tokenizer.pad_token_id, max_steps=cfg.optim.max_steps, lr=cfg.optim.lr,
            warmup_ratio=cfg.optim.warmup_ratio, max_grad_norm=cfg.optim.max_grad_norm, micro_batch=cfg.optim.micro_batch,
            device=device, run_dir=out_dir, eval_every=eval_every, eval_fn=eval_fn,
        )
        status = train_result.status
        metrics["status"] = status
        metrics["train_gpu_seconds"] = train_result.train_gpu_seconds
        metrics["train_wall_seconds"] = train_result.train_wall_seconds
        metrics["peak_vram_mb"] = train_result.peak_vram_mb
        metrics["samples_per_sec"] = (
            (train_result.final_step * cfg.optim.micro_batch) / train_result.train_wall_seconds
            if train_result.train_wall_seconds > 0 else 0.0
        )
        if train_result.held_out_loss_curve:
            final_curve_point = train_result.held_out_loss_curve[-1]
            metrics["loss_token_weighted"] = final_curve_point["loss_token_weighted"]
            metrics["loss_example_mean"] = final_curve_point["loss_example_mean"]
        (out_dir / "held_out_loss_curve.json").write_text(json.dumps(train_result.held_out_loss_curve, indent=2))
        model.eval()

    if status == "ok":
        eval_gpu_total = 0.0
        if metrics["loss_token_weighted"] is None:  # zero_shot: no periodic curve, evaluate once here
            held_out, secs = gpu_time(device, lambda: compute_held_out_loss(
                model, bundle.val_loss, tokenizer.pad_token_id, device, micro_batch=cfg.optim.micro_batch, use_amp=use_amp
            ))
            metrics["loss_token_weighted"] = held_out["loss_token_weighted"]
            metrics["loss_example_mean"] = held_out["loss_example_mean"]
            eval_gpu_total += secs

        if cfg.eval.run_generation and bundle.val_gen:
            gen_result, secs = gpu_time(device, lambda: compute_gsm8k_generation(
                model, tokenizer, bundle.val_gen, device, micro_batch=cfg.optim.micro_batch, max_new_tokens=cfg.eval.max_new_tokens
            ))
            eval_gpu_total += secs
            metrics["gsm8k_strict"] = gen_result["gsm8k_strict"]
            metrics["gsm8k_flexible"] = gen_result["gsm8k_flexible"]
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "samples.jsonl").open("w") as fh:
                for sample in gen_result["samples"]:
                    fh.write(json.dumps(sample) + "\n")
        metrics["eval_gpu_seconds"] = eval_gpu_total

        if cfg.alloc.strategy != "zero_shot":
            model.save_pretrained(str(out_dir / "adapter"))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    atomic_append_csv(Path("results/results.csv"), metrics, RESULT_FIELDS)

    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
