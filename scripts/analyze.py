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


def compare_to_uniform(df: pd.DataFrame, strategy: str, metric: str, sd: float) -> dict:
    """sd is the noise-floor sigma (pooled sd of the uniform-baseline seeds), shared across every
    comparison for this metric -- but the MDE itself is computed per comparison, using that
    comparison's own paired n, not the uniform baseline's seed count. Tier-1's grid deliberately runs
    unequal seed counts (5 for uniform/gradnorm_prop, 3 for the two controls, BUILD_SPEC.md §6), and
    a 3-seed comparison has a coarser (larger) true detection threshold than a 5-seed one -- applying
    one blanket MDE to every comparison understated that for the smaller-n arms.
    """
    uniform = by_seed(df, "uniform", metric)
    other = by_seed(df, strategy, metric)
    result = paired_delta(other, uniform)
    mde = minimum_detectable_effect(sd, result.n) if result.n >= 2 else float("nan")
    return {
        "strategy": strategy,
        "metric": metric,
        "mean_delta_vs_uniform": result.mean_delta,
        "n_pairs": result.n,
        "minimum_detectable_effect": mde,
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


def interpretation(comparisons: dict, lower_is_better: bool) -> str:
    """BUILD_SPEC.md §4.4's four-outcome table, extended with a fifth outcome the table doesn't name:
    every non-uniform strategy reliably *worse* than uniform. With n<=5 seeds this is a coarse
    mean-delta-vs-MDE read, not a statistical test -- this pipeline deliberately computes no
    significance tests anywhere (§4.8); a human should read the raw `comparisons` alongside this
    label, not trust it blindly. Each comparison is gated on its *own* MDE (its own paired n), not a
    single blanket threshold -- tier 1's grid runs unequal seed counts by design (§6), so a 3-seed
    control has a coarser true detection threshold than the 5-seed hypothesis arm.
    """
    directions = {}
    for strat in CONTROL_STRATEGIES:
        cmp = comparisons.get(strat)
        if cmp is None:
            return f"insufficient conditions to interpret (missing {strat} vs uniform)"
        directions[strat] = _direction(cmp["mean_delta_vs_uniform"], cmp["minimum_detectable_effect"], lower_is_better)
    if any(v is None for v in directions.values()):
        return "insufficient seeds for an MDE-gated read on at least one comparison"

    prop, inverse, random_ = directions["gradnorm_prop"], directions["gradnorm_inverse"], directions["random"]
    mdes = ", ".join(f"{s}={comparisons[s]['minimum_detectable_effect']:.4g}" for s in CONTROL_STRATEGIES)

    if prop == "worse" and inverse == "worse" and random_ == "worse":
        return (
            f"uniform beats every non-uniform strategy tested, each beyond its own MDE ({mdes}) -- "
            "non-uniform allocation is actively worse here, not neutral; this is a real negative "
            "result, not 'allocation doesn't matter'"
        )
    if prop == "noise" and inverse == "noise" and random_ == "noise":
        return f"prop ~ inverse ~ random ~ uniform: allocation doesn't matter at this scale (MDEs: {mdes})"
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
    # reference MDE at the uniform baseline's own seed count -- a headline number, not what every
    # comparison is actually gated on (see compare_to_uniform / interpretation for the per-comparison
    # MDEs that unequal-n comparisons, e.g. tier 1's 3-seed controls, actually need).
    mde_at_n_uniform = minimum_detectable_effect(sd, n_uniform) if n_uniform >= 2 else float("nan")

    comparisons = {}
    for strategy in df["strategy"].unique():
        if strategy in ("uniform", "zero_shot"):
            continue
        comparisons[strategy] = compare_to_uniform(df, strategy, metric, sd)

    return {
        "noise_floor_pooled_sd": sd,
        "n_uniform_seeds": n_uniform,
        "minimum_detectable_effect_at_n_uniform": mde_at_n_uniform,
        "comparisons": comparisons,
        "interpretation": interpretation(comparisons, LOWER_IS_BETTER[metric])
        if not pd.isna(sd) else "insufficient uniform seeds for MDE",
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
