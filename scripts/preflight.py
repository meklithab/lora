"""Phase 0 preflight: verify CUDA, download assets, calibrate micro-batch, project tier-1 GPU-hours.

Run on the actual Kaggle/Colab T4 or P100 session before any training. The projection this prints
gates whether tier 1 runs as specified or gets cut down (BUILD_SPEC.md §8, P0) -- that decision
belongs to the repo owner, not to this script.
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_SEQ_LEN = 512
REFERENCE_RANK = 16  # tier-1 budget_rank, BUILD_SPEC.md §6
ALPHA_RATIO = 2  # constant_ratio scaling, BUILD_SPEC.md §4.5
TIMING_STEPS = 20
CANDIDATE_MICRO_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]

# Tier-1 workload, BUILD_SPEC.md §6 / §8
TIER1_TRAIN_RUNS = 16  # uniform x5 + gradnorm_prop x5 + gradnorm_inverse x3 + random x3
TIER1_MAX_STEPS = 400
TIER1_PROBES = 2  # gsm8k, alpaca
PROBE_STEPS = 100
EVAL_LOSS_EXAMPLES = 300
EVAL_GEN_EXAMPLES = 200
EVAL_GEN_RUNS = 17  # 16 trained + 1 zero-shot reference


def assert_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Preflight targets a T4/P100 Kaggle/Colab GPU runtime.")
    name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name} ({total_vram_gb:.1f} GB)")
    return name, total_vram_gb


def download_assets():
    print(f"Downloading model: {MODEL_NAME}")
    AutoTokenizer.from_pretrained(MODEL_NAME)
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, attn_implementation="sdpa")
    print("Downloading datasets: openai/gsm8k (main), tatsu-lab/alpaca")
    load_dataset("openai/gsm8k", "main")
    load_dataset("tatsu-lab/alpaca")


def build_reference_model():
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).cuda()
    vocab_size = base.config.vocab_size
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    lora_cfg = LoraConfig(
        r=REFERENCE_RANK,
        lora_alpha=ALPHA_RATIO * REFERENCE_RANK,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    return model, vocab_size


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _train_step(model, optimizer, scaler, input_ids, labels):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(input_ids=input_ids, labels=labels)
    scaler.scale(out.loss).backward()
    scaler.step(optimizer)
    scaler.update()


def try_micro_batch(bs):
    model, vocab_size = build_reference_model()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    input_ids = torch.randint(0, vocab_size, (bs, MAX_SEQ_LEN), device="cuda")
    labels = input_ids.clone()
    try:
        for _ in range(2):
            _train_step(model, optimizer, scaler, input_ids, labels)
        torch.cuda.synchronize()
        return True
    except RuntimeError as exc:
        if _is_oom(exc):
            return False
        raise
    finally:
        del model, optimizer, scaler
        torch.cuda.empty_cache()


def calibrate_micro_batch():
    fitting = None
    for bs in CANDIDATE_MICRO_BATCHES:
        print(f"Trying micro-batch size {bs}...")
        if try_micro_batch(bs):
            fitting = bs
        else:
            print(f"  OOM at {bs}")
            break
    if fitting is None:
        raise RuntimeError("Even micro-batch size 1 OOMs at max_seq_len=512.")
    if fitting == CANDIDATE_MICRO_BATCHES[-1]:
        print(f"Largest candidate ({fitting}) still fit -- raise CANDIDATE_MICRO_BATCHES to probe higher.")
    print(f"Largest fitting micro-batch: {fitting} (train.py should use this with no runtime headroom cut)")
    return fitting


def time_steps(micro_batch, n_steps=TIMING_STEPS):
    model, vocab_size = build_reference_model()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    input_ids = torch.randint(0, vocab_size, (micro_batch, MAX_SEQ_LEN), device="cuda")
    labels = input_ids.clone()

    for _ in range(2):
        _train_step(model, optimizer, scaler, input_ids, labels)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(n_steps):
        _train_step(model, optimizer, scaler, input_ids, labels)
    end_evt.record()
    torch.cuda.synchronize()

    gpu_seconds = start_evt.elapsed_time(end_evt) / 1000.0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return gpu_seconds / n_steps, peak_vram_mb


def project_tier1(per_step_seconds):
    """Coarse GPU-hour projection. Train/probe timing is measured directly; eval timing is a rough
    multiplier on the measured train step, not a separate simulation -- good enough to gate the
    max_steps/8h decision in BUILD_SPEC.md §8, not precise to the minute.
    """
    train_seconds = TIER1_TRAIN_RUNS * TIER1_MAX_STEPS * per_step_seconds
    probe_seconds = TIER1_PROBES * PROBE_STEPS * per_step_seconds
    eval_loss_seconds = EVAL_GEN_RUNS * EVAL_LOSS_EXAMPLES * (per_step_seconds / 3)
    eval_gen_seconds = EVAL_GEN_RUNS * EVAL_GEN_EXAMPLES * (per_step_seconds * 2)
    total_seconds = train_seconds + probe_seconds + eval_loss_seconds + eval_gen_seconds
    return {
        "train_hours": train_seconds / 3600,
        "probe_hours": probe_seconds / 3600,
        "eval_loss_hours": eval_loss_seconds / 3600,
        "eval_gen_hours": eval_gen_seconds / 3600,
        "total_hours": total_seconds / 3600,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/preflight.json")
    args = parser.parse_args()

    gpu_name, vram_gb = assert_cuda()
    download_assets()
    micro_batch = calibrate_micro_batch()
    per_step_seconds, peak_vram_mb = time_steps(micro_batch)
    projection = project_tier1(per_step_seconds)

    print(f"\nPer-step time at micro-batch={micro_batch}: {per_step_seconds * 1000:.1f} ms")
    print(f"Peak VRAM at timing batch: {peak_vram_mb:.0f} MB")
    print("\nTier-1 GPU-hour projection (coarse eval estimate, see project_tier1 docstring):")
    for key, hours in projection.items():
        print(f"  {key}: {hours:.2f} h")

    result = {
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "micro_batch": micro_batch,
        "per_step_seconds": per_step_seconds,
        "peak_vram_mb": peak_vram_mb,
        "tier1_projection_hours": projection,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
