"""Paired deltas, noise floor, MDE, effect sizes, and the four-outcome interpretation table from
BUILD_SPEC.md §4.4. Reads results.csv only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from rankalloc.metrics import cliffs_delta, hedges_g, minimum_detectable_effect, paired_delta, pooled_sd

PRIMARY_METRIC = "loss_token_weighted"  # lower is better
SECONDARY_METRIC = "gsm8k_flexible"  # higher is better
LOWER_IS_BETTER = {"loss_token_weighted": True, "gsm8k_flexible": False}
CONTROL_STRATEGIES = ("gradnorm_prop", "gradnorm_inverse", "random")


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["status"] == "ok"]


def by_seed(df: pd.DataFrame, strategy: str, metric: str) -> dict:
    sub = df[df["strategy"] == strategy]
    return dict(zip(sub["seed"], sub[metric]))


def compare_to_uniform(df: pd.DataFrame, strategy: str, metric: str) -> dict:
    uniform = by_seed(df, "uniform", metric)
    other = by_seed(df, strategy, metric)
    result = paired_delta(other, uniform)
    return {
        "strategy": strategy,
        "metric": metric,
        "mean_delta_vs_uniform": result.mean_delta,
        "n_pairs": result.n,
        "pairs": result.pairs,
        "hedges_g": hedges_g(list(other.values()), list(uniform.values())),
        "cliffs_delta": cliffs_delta(list(other.values()), list(uniform.values())),
    }


def _direction(delta, mde, lower_is_better):
    """'better' / 'worse' / 'noise' relative to uniform, gated on the MDE -- not just "does it beat
    uniform": a strategy can be reliably *worse* than uniform (beyond the noise floor, wrong
    direction), which is a distinct, reportable finding from "no detectable effect either way". The
    original two-state (beats-uniform yes/no) version of this function collapsed both into the same
    "doesn't matter" label, which is wrong -- a clean negative result isn't a null result.
    """
    if delta is None or mde is None or pd.isna(delta) or pd.isna(mde):
        return None
    improvement = -delta if lower_is_better else delta
    if improvement > mde:
        return "better"
    if improvement < -mde:
        return "worse"
    return "noise"


def interpretation(comparisons: dict, mde: float, lower_is_better: bool) -> str:
    """BUILD_SPEC.md §4.4's four-outcome table, extended with a fifth outcome the table doesn't name:
    every non-uniform strategy reliably *worse* than uniform. With n<=5 seeds this is a coarse
    mean-delta-vs-MDE read, not a statistical test -- this pipeline deliberately computes no
    significance tests anywhere (§4.8); a human should read the raw `comparisons` alongside this
    label, not trust it blindly.
    """
    directions = {}
    for strat in CONTROL_STRATEGIES:
        cmp = comparisons.get(strat)
        if cmp is None:
            return f"insufficient conditions to interpret (missing {strat} vs uniform)"
        directions[strat] = _direction(cmp["mean_delta_vs_uniform"], mde, lower_is_better)
    if any(v is None for v in directions.values()):
        return "insufficient uniform seeds for an MDE-gated read"

    prop, inverse, random_ = directions["gradnorm_prop"], directions["gradnorm_inverse"], directions["random"]

    if prop == "worse" and inverse == "worse" and random_ == "worse":
        return (
            f"uniform beats every non-uniform strategy tested, beyond the MDE (MDE={mde:.4g}) -- "
            "non-uniform allocation is actively worse here, not neutral; this is a real negative "
            "result, not 'allocation doesn't matter'"
        )
    if prop == "noise" and inverse == "noise" and random_ == "noise":
        return f"prop ~ inverse ~ random ~ uniform: allocation doesn't matter at this scale (MDE={mde:.4g})"
    if prop == "better" and inverse == "better" and random_ == "better":
        return "prop ~ inverse ~ random > uniform: non-uniformity itself helps; the gradient signal is irrelevant"
    if prop == "better" and inverse == "better" and random_ != "better":
        return "prop ~ inverse > random: the signal identifies *something*, but direction doesn't matter -- suspect a width/position correlate"
    if prop == "better" and inverse != "better":
        return "prop > random > inverse: the gradient signal is informative AND directional -- the strong result"
    return (
        f"mixed directions (prop={prop}, inverse={inverse}, random={random_}) -- no clean match to a "
        "canonical outcome, inspect the raw per-strategy deltas"
    )


def analyze_metric(df: pd.DataFrame, metric: str) -> dict:
    uniform_values = list(by_seed(df, "uniform", metric).values())
    n_uniform = len(uniform_values)
    sd = pooled_sd(uniform_values) if n_uniform >= 2 else float("nan")
    mde = minimum_detectable_effect(sd, n_uniform) if n_uniform >= 2 else float("nan")

    comparisons = {}
    for strategy in df["strategy"].unique():
        if strategy in ("uniform", "zero_shot"):
            continue
        comparisons[strategy] = compare_to_uniform(df, strategy, metric)

    return {
        "noise_floor_pooled_sd": sd,
        "n_uniform_seeds": n_uniform,
        "minimum_detectable_effect": mde,
        "comparisons": comparisons,
        "interpretation": interpretation(comparisons, mde, LOWER_IS_BETTER[metric])
        if not pd.isna(mde) else "insufficient uniform seeds for MDE",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/results.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/analysis.json"))
    args = parser.parse_args()

    df = load_results(args.results)
    output = {metric: analyze_metric(df, metric) for metric in (PRIMARY_METRIC, SECONDARY_METRIC)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, default=str))
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
