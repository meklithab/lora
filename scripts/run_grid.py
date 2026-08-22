"""Resumable grid driver. BUILD_SPEC.md §5, §8 P5.

Each run executes as an isolated `run_single.py` (or `run_probe.py`) subprocess: crash/OOM
containment per condition, and CUDA memory is released by process exit rather than accumulating
across a 16+ run grid in one long-lived process. `results.csv` itself is written by run_single.py's
own atomic append; this driver decides what to run and what to skip, and writes a synthetic
`status=failed` row only when a subprocess dies before it could write its own row.
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rankalloc.config import RunConfig, apply_overrides, from_yaml
from rankalloc.config import run_id as compute_run_id
from rankalloc.io_utils import RESULT_FIELDS, RESULTS_CSV, atomic_append_csv, existing_run_ids, run_dir as get_run_dir
from rankalloc.probe import compute_probe_id

GRADNORM_STRATEGIES = ("gradnorm_prop", "gradnorm_inverse")
ALLOCATION_DRIVING_TASK = "gsm8k"  # the only task any allocation strategy actually consumes signal from

FRIENDLY_KEY_MAP = {
    "budget_rank": "alloc.budget_rank",
    "scaling_mode": "scaling.mode",
    "signal": "alloc.signal",
    "max_steps": "optim.max_steps",
    "temperature": "alloc.temperature",
    "strategy": "alloc.strategy",
}


def load_grid(path: Path) -> dict:
    with open(path) as fh:
        spec = yaml.safe_load(fh)
    required = {"budget_rank", "scaling_mode", "signal", "max_steps", "conditions"}
    unknown = set(spec) - (required | {"probes", "probe_seeds"})
    if unknown:
        raise ValueError("grid config " + str(path) + " has unknown key(s): " + str(sorted(unknown)))
    missing = required - set(spec)
    if missing:
        raise ValueError(f"grid config {path} missing required keys: {sorted(missing)}")
    return spec


def _to_override(key: str, value) -> str:
    dotted = FRIENDLY_KEY_MAP.get(key, key)
    return f"{dotted}={value}"


def expand_grid(spec: dict, base_config_path: Path) -> List[dict]:
    """Return a list of {condition, seed, train, overrides} dicts, one per (condition, seed, and any
    swept per-condition scalar list value, e.g. temperature: [0.5, 2.0])."""
    grid_level = {k: spec[k] for k in ("budget_rank", "scaling_mode", "signal", "max_steps")}
    runs = []
    for cond in spec["conditions"]:
        strategy = cond["strategy"]
        n_seeds = cond["n_seeds"]
        train = cond.get("train", True)
        cond_overrides = {k: v for k, v in cond.items() if k not in ("strategy", "n_seeds", "train")}

        swept_keys = [k for k, v in cond_overrides.items() if isinstance(v, list)]
        scalar_overrides = {k: v for k, v in cond_overrides.items() if k not in swept_keys}
        sweep_value_combos = list(itertools.product(*[cond_overrides[k] for k in swept_keys])) or [()]

        for combo in sweep_value_combos:
            swept = dict(zip(swept_keys, combo))
            merged = {**grid_level, **scalar_overrides, **swept}
            overrides = [_to_override(k, v) for k, v in merged.items()] + [_to_override("strategy", strategy)]
            for seed in range(n_seeds):
                runs.append({
                    "condition": strategy,
                    "seed": seed,
                    "train": train,
                    "overrides": overrides + [f"seed={seed}"],
                })
    return runs


def resolve_config(base_config_path: Path, overrides: List[str]) -> RunConfig:
    cfg = from_yaml(base_config_path) if base_config_path else RunConfig()
    return apply_overrides(cfg, overrides)


def probe_seeds(spec: dict) -> List[int]:
    """Seeds to draw the probe at. Several seeds make the *allocation* a replicated draw rather
    than a single fixed one, which is what the statistical unit has to be for a claim about the
    probe->allocation procedure (as opposed to a claim about one particular allocation).
    """
    return [int(x) for x in spec.get("probe_seeds", [0])]


def needed_probe_tasks(spec: dict, runs: List[dict]) -> List[str]:
    tasks = set(spec.get("probes", []))
    if any(r["condition"] in GRADNORM_STRATEGIES for r in runs):
        tasks.add(ALLOCATION_DRIVING_TASK)
    return sorted(tasks)


def probe_id_for(base_cfg: RunConfig, task: str, seed: int) -> str:
    return compute_probe_id(
        base_cfg.model_name, task, base_cfg.probe.rank, base_cfg.probe.steps,
        seed=seed, split_seed=base_cfg.data.split_seed, freeze_a=base_cfg.probe.freeze_a,
    )


def run_probes(tasks, seeds, base_config_path: Path, device: str, dry_run: bool) -> Dict[tuple, str]:
    probe_ids = {}
    base_cfg = from_yaml(base_config_path) if base_config_path else RunConfig()
    for task in tasks:
        for seed in seeds:
            probe_id = probe_id_for(base_cfg, task, seed)
            probe_ids[(task, seed)] = probe_id
            probe_path = Path("results/probe") / (probe_id + ".json")
            if probe_path.exists():
                print("skip probe " + probe_id + " (" + task + ", seed=" + str(seed) + "): already cached")
                continue
            if dry_run:
                print(probe_id + "  probe:" + task + " seed=" + str(seed) + "  RUN")
                continue
            cmd = [sys.executable, "scripts/run_probe.py", "--device", device,
                   "--set", "probe.task=" + task, "--set", "seed=" + str(seed)]
            print("running probe " + probe_id + " (" + task + ", seed=" + str(seed) + ")")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not probe_path.exists():
                err_dir = Path("results/runs") / ("probe-" + probe_id)
                err_dir.mkdir(parents=True, exist_ok=True)
                (err_dir / "error.log").write_text(
                    "cmd: " + " ".join(cmd) + chr(10) * 2
                    + "STDOUT:" + chr(10) + result.stdout + chr(10) * 2
                    + "STDERR:" + chr(10) + result.stderr + chr(10)
                )
                print("FAILED probe " + probe_id + ": see " + str(err_dir / "error.log"))
    return probe_ids


def with_probe_override(run: dict, seeds, probe_ids) -> List[str]:
    """Append the alloc.probe_id override a gradnorm run needs, if any.

    Shared by the dry run and the real run on purpose. run_id hashes the fully resolved config
    including alloc.probe_id, so a dry run that skipped this override reported run_ids that no real
    run would ever produce -- making its SKIP/RUN resume column silently wrong for exactly the arms
    that depend on the probe.

    The run's seed is rotated across the available probe draws so the *allocation* is replicated
    rather than fixed. With a single probe seed this reduces to the previous behaviour.
    """
    overrides = list(run["overrides"])
    if run["condition"] in GRADNORM_STRATEGIES:
        pseed = seeds[run["seed"] % len(seeds)]
        probe_id = probe_ids.get((ALLOCATION_DRIVING_TASK, pseed))
        if not probe_id:
            raise AssertionError(
                f"strategy={run['condition']!r} needs the {ALLOCATION_DRIVING_TASK} probe at "
                f"seed={pseed}, which is not available"
            )
        overrides.append(f"alloc.probe_id={probe_id}")
    return overrides


def estimate_run_seconds(cfg: RunConfig, train: bool, per_step_seconds: float) -> float:
    """Coarse projection, same methodology as preflight.py's project_tier1 -- a multiplier on the
    measured train-step time, not a full simulation. Good enough to gate --max_hours, not exact."""
    eval_seconds = cfg.data.val_loss_n * (per_step_seconds / 3) + cfg.data.val_gen_n * (per_step_seconds * 2)
    if not train:
        return eval_seconds
    return cfg.optim.max_steps * per_step_seconds + eval_seconds


def load_calibration() -> float:
    path = Path("results/preflight.json")
    if not path.exists():
        print("WARNING: results/preflight.json not found -- run scripts/preflight.py for an accurate "
              "--dry_run projection. Using a conservative fallback of 0.5s/step.")
        return 0.5
    return json.loads(path.read_text())["per_step_seconds"]


def write_failed_row(run_id: str, cond: dict, cfg: RunConfig):
    row = {field: "" for field in RESULT_FIELDS}
    row.update(
        run_id=run_id, condition=cond["condition"], seed=cond["seed"], strategy=cfg.alloc.strategy,
        temperature=cfg.alloc.temperature, scaling_mode=cfg.scaling.mode, budget_rank=cfg.alloc.budget_rank,
        max_steps=cfg.optim.max_steps, status="failed",
    )
    atomic_append_csv(RESULTS_CSV, row, RESULT_FIELDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--base_config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_hours", type=float, default=None)
    args = parser.parse_args()

    spec = load_grid(args.grid)
    runs = expand_grid(spec, args.base_config)
    probe_tasks = needed_probe_tasks(spec, runs)
    seeds_for_probes = probe_seeds(spec)

    if args.dry_run:
        per_step_seconds = load_calibration()
        probe_hours = 0.0
        base_cfg = from_yaml(args.base_config) if args.base_config else RunConfig()
        for task in probe_tasks:
            for pseed in seeds_for_probes:
                probe_id = probe_id_for(base_cfg, task, pseed)
                cached = (Path("results/probe") / f"{probe_id}.json").exists()
                state = "SKIP (cached)" if cached else "RUN"
                print(f"{probe_id}  probe:{task} seed={pseed}  {state}")
                if not cached:
                    probe_hours += (base_cfg.probe.steps * per_step_seconds) / 3600.0

        completed = existing_run_ids(RESULTS_CSV, exclude_statuses={"failed"})
        dry_probe_ids = {
            (task, pseed): probe_id_for(base_cfg, task, pseed)
            for task in probe_tasks
            for pseed in seeds_for_probes
        }
        condition_hours = 0.0
        for r in runs:
            cfg = resolve_config(args.base_config, with_probe_override(r, seeds_for_probes, dry_probe_ids))
            rid = compute_run_id(cfg)
            done = rid in completed
            print(f"{rid}  {r['condition']:16s} seed={r['seed']}  {'SKIP (done)' if done else 'RUN'}")
            if not done:
                condition_hours += estimate_run_seconds(cfg, r["train"], per_step_seconds) / 3600.0
        total = probe_hours + condition_hours
        n_probes = len(probe_tasks) * len(seeds_for_probes)
        print(
            f"\n{n_probes} probe(s) ({len(probe_tasks)} task(s) x {len(seeds_for_probes)} seed(s)), "
            f"{len(runs)} condition-runs; projected {total:.2f} GPU-hours for pending work"
        )
        return

    probe_ids = run_probes(probe_tasks, seeds_for_probes, args.base_config, args.device, dry_run=False)

    completed = existing_run_ids(RESULTS_CSV, exclude_statuses={"failed"})
    hours_spent = 0.0
    for r in runs:
        overrides = with_probe_override(r, seeds_for_probes, probe_ids)
        cfg = resolve_config(args.base_config, overrides)
        rid = compute_run_id(cfg)

        if rid in completed:
            print(f"skip {rid} ({r['condition']} seed={r['seed']}): already in results.csv")
            continue
        if args.max_hours is not None and hours_spent >= args.max_hours:
            print(f"stopping: max_hours={args.max_hours} reached ({hours_spent:.2f}h spent)")
            break

        cmd = [sys.executable, "scripts/run_single.py", "--device", args.device, "--condition", r["condition"]]
        if args.base_config:
            cmd += ["--config", str(args.base_config)]
        for ov in overrides:
            cmd += ["--set", ov]

        print(f"running {rid} ({r['condition']} seed={r['seed']})")
        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        hours_spent += (time.perf_counter() - start) / 3600.0

        if result.returncode != 0:
            run_dir = get_run_dir(rid)
            (run_dir / "error.log").write_text(f"cmd: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n")
            print(f"FAILED {rid}: see {run_dir / 'error.log'}")
            if rid not in existing_run_ids(RESULTS_CSV):
                write_failed_row(rid, r, cfg)
            continue
        completed.add(rid)

    print(f"\ndone: {hours_spent:.2f} GPU-hours spent this invocation")


if __name__ == "__main__":
    main()
