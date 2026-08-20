"""Six figures, each regenerable from the results/ tree alone (no re-running experiments).
BUILD_SPEC.md §5.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rankalloc.metrics import minimum_detectable_effect, paired_delta, pooled_sd

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
PRIMARY_METRIC = "loss_token_weighted"


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "results.csv")
    return df[df["status"] == "ok"]


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_allocation_profiles():
    """Rank/signal vs layer index, GSM8K vs Alpaca side by side -- built from the probe JSON files
    directly (results.csv has no per-module data), since this is what §4.3 means by "comparing the
    two allocation profiles".
    """
    probe_dir = RESULTS_DIR / "probe"
    by_task = {}
    for p in sorted(probe_dir.glob("*.json")) if probe_dir.exists() else []:
        data = json.loads(p.read_text())
        by_task.setdefault(data["task"], data)  # first probe found per task

    tasks = [t for t in ("gsm8k", "alpaca") if t in by_task]
    if not tasks:
        print("skip fig_allocation_profiles: no probe JSON under results/probe/")
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4), sharey=True)
    axes = [axes] if len(tasks) == 1 else axes
    for ax, task in zip(axes, tasks):
        data = by_task[task]
        rms, meta = data["signals"]["rms"], data["module_meta"]
        ax.scatter([meta[m]["layer_idx"] for m in rms], list(rms.values()), s=8, alpha=0.6)
        ax.set_title(f"{task} probe rms signal")
        ax.set_xlabel("layer index")
    axes[0].set_ylabel("rms gradient signal")
    fig.suptitle("Allocation-driving signal by layer (probe rms)")
    _save(fig, "01_allocation_profiles.png")


def fig_forest_plot(df, metric=PRIMARY_METRIC):
    uniform = dict(zip(df[df.strategy == "uniform"]["seed"], df[df.strategy == "uniform"][metric]))
    n_uniform = len(uniform)
    sd = pooled_sd(list(uniform.values())) if n_uniform >= 2 else float("nan")
    mde = minimum_detectable_effect(sd, n_uniform) if n_uniform >= 2 else float("nan")

    strategies = [s for s in df["strategy"].unique() if s not in ("uniform", "zero_shot")]
    if not strategies:
        print("skip fig_forest_plot: no non-uniform conditions in results.csv yet")
        return

    fig, ax = plt.subplots(figsize=(7, 1.2 * len(strategies) + 1))
    for i, strat in enumerate(strategies):
        other = dict(zip(df[df.strategy == strat]["seed"], df[df.strategy == strat][metric]))
        result = paired_delta(other, uniform)
        ax.scatter([p["delta"] for p in result.pairs], [i] * len(result.pairs), color="gray", alpha=0.6, zorder=2)
        if result.n:
            ax.scatter([result.mean_delta], [i], color="C0", marker="D", s=60, zorder=3)
    if not pd.isna(mde):
        ax.axvspan(-mde, mde, color="C1", alpha=0.15, label=f"MDE band (+/-{mde:.3g})")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)
    ax.set_xlabel(f"paired delta vs uniform ({metric})")
    ax.legend()
    fig.suptitle("Forest plot: paired delta vs uniform, individual seeds + MDE band")
    _save(fig, "02_forest_plot.png")


def fig_learning_curves(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for run_id in df["run_id"]:
        curve_path = RESULTS_DIR / "runs" / str(run_id) / "held_out_loss_curve.json"
        if not curve_path.exists():
            continue
        curve = json.loads(curve_path.read_text())
        step_log_path = RESULTS_DIR / "runs" / str(run_id) / "step_log.jsonl"
        gpu_by_step = {}
        if step_log_path.exists():
            for line in step_log_path.read_text().splitlines():
                row = json.loads(line)
                gpu_by_step[row["step"] + 1] = row["cumulative_gpu_seconds"]
        xs = [gpu_by_step.get(p["step"], p["step"]) for p in curve]
        ys = [p["loss_token_weighted"] for p in curve]
        condition = df.loc[df["run_id"] == run_id, "condition"].iloc[0]
        ax.plot(xs, ys, marker="o", markersize=3, alpha=0.7, label=f"{condition} ({run_id})")
        plotted = True
    if not plotted:
        print("skip fig_learning_curves: no held_out_loss_curve.json files found under results/runs/")
        plt.close(fig)
        return
    ax.set_xlabel("cumulative GPU-seconds")
    ax.set_ylabel("held-out loss (token-weighted)")
    ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Learning curves: held-out loss vs GPU-seconds")
    _save(fig, "03_learning_curves.png")


def fig_noise_floor(df, metric=PRIMARY_METRIC):
    uniform = df[df.strategy == "uniform"][metric].dropna().tolist()
    if len(uniform) < 2:
        print("skip fig_noise_floor: fewer than 2 uniform-baseline seeds in results.csv")
        return
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.scatter([0] * len(uniform), uniform, alpha=0.7, zorder=2)
    mean, sd = sum(uniform) / len(uniform), pooled_sd(uniform)
    ax.errorbar([0], [mean], yerr=[sd], fmt="D", color="C1", capsize=6, zorder=3, label=f"mean +/- sd ({sd:.3g})")
    ax.set_xticks([])
    ax.set_ylabel(metric)
    ax.legend()
    fig.suptitle(f"Noise floor: {len(uniform)}-seed uniform-baseline spread")
    _save(fig, "04_noise_floor.png")


def fig_scaling_trap(df, metric=PRIMARY_METRIC):
    constant = df[df.scaling_mode == "constant_ratio"]
    fixed = df[df.scaling_mode == "fixed_alpha"]
    if constant.empty or fixed.empty:
        print("skip fig_scaling_trap: need both constant_ratio and fixed_alpha rows in results.csv")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, sub in (("constant_ratio", constant), ("fixed_alpha", fixed)):
        ax.scatter(sub["strategy"], sub[metric], label=label, alpha=0.7)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.suptitle("Scaling-trap panel: constant_ratio vs fixed_alpha at identical allocations")
    _save(fig, "05_scaling_trap.png")


def fig_budget_fidelity(df):
    sub = df.dropna(subset=["budget_rel_error"])
    sub = sub[sub["strategy"] != "zero_shot"]
    if sub.empty:
        print("skip fig_budget_fidelity: no allocation-driven rows in results.csv")
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(sub)), 4))
    ax.bar(range(len(sub)), sub["budget_rel_error"] * 100, tick_label=sub["run_id"])
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="0.5% target ceiling")
    ax.set_ylabel("budget_rel_error (%)")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.legend()
    fig.suptitle("Budget fidelity: realised vs target parameter count (proof of I1)")
    _save(fig, "06_budget_fidelity.png")


def main():
    df = load_results()
    fig_allocation_profiles()
    fig_forest_plot(df)
    fig_learning_curves(df)
    fig_noise_floor(df)
    fig_scaling_trap(df)
    fig_budget_fidelity(df)


if __name__ == "__main__":
    main()
