"""CLI: run the gradient-sensitivity probe from a config file. BUILD_SPEC.md §4.3."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from rankalloc.config import RunConfig, apply_overrides, from_yaml
from rankalloc.probe import run_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("results/probe"))
    args = parser.parse_args()

    cfg = from_yaml(args.config) if args.config else RunConfig()
    cfg = apply_overrides(cfg, args.overrides)

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    result = run_probe(
        model_name=cfg.model_name,
        task=cfg.probe.task,
        rank=cfg.probe.rank,
        steps=cfg.probe.steps,
        lr=cfg.optim.lr,
        warmup_ratio=cfg.optim.warmup_ratio,
        seed=cfg.seed,
        split_seed=cfg.data.split_seed,
        max_seq_len=cfg.data.max_seq_len,
        micro_batch=cfg.optim.micro_batch,
        alpha_ratio=cfg.scaling.alpha_ratio,
        target_modules=cfg.target_modules,
        device=args.device,
        dtype=dtype,
        out_dir=args.out_dir,
    )
    print(f"probe_id={result.probe_id} task={result.task} wall={result.probe_wall_seconds:.1f}s gpu={result.probe_gpu_seconds:.1f}s")


if __name__ == "__main__":
    main()
