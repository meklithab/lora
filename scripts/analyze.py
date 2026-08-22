"""Paired deltas, noise floor, MDE, effect sizes, and the four-outcome interpretation table from
BUILD_SPEC.md 4.4. Reads results.csv only.

Two filtering rules matter here and are enforced rather than assumed:

  - rows are grouped by a full *arm key*, not by `strategy` alone. Tier 2 runs several
    gradnorm_prop variants (fixed_alpha, rslora, signal=raw_norm, temperature sweeps) that all
    write strategy='gradnorm_prop' with seeds 0..2; grouping on strategy alone silently blends them
    into the tier-1 arm and, because the per-seed lookup is a dict, keeps whichever row happens to
    land last in the file.
  - non-experimental rows (smoke tests) are dropped, and duplicate (arm, seed) pairs are a hard
    error rather than a silent last-write-wins.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from rankalloc.metrics import (
    cliffs_delta,
    hedges_g,
    minimum_detectable_effect,
    paired_delta,
    pooled_sd,
    pooled_two_sample_sd,
    sign_flip_test,
)

PRIMARY_METRIC = "loss_token_weighted"  # lower is better
SECONDARY_METRIC = "gsm8k_flexible"  # higher is better
LOWER_IS_BETTER = {"loss_token_weighted": True, "gsm8k_flexible": False}
CONTROL_STRATEGIES = ("gradnorm_prop", "gradnorm_inverse", "random")

# Fields that define a distinct experimental arm. Two rows sharing all of these and a seed are a
# duplicate, not two observations.
ARM_FIELDS = ("strategy", "signal", "temperature", "scaling_mode", "budget_rank", "max_steps")
# Fields the baseline must match for a comparison to be like-for-like. `signal` and `temperature`
# are excluded: they are meaningless for the uniform strategy, which consumes no probe.
PROTOCOL_FIELDS = ("scaling_mode", "budget_rank", "max_steps")

EXCLUDED_CONDITIONS = ("smoke_test",)


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df = df[~df["condition"].isin(EXCLUDED_CONDITIONS)]
    for col in ("signal", "temperature", "scaling_mode", "budget_rank", "max_steps"):
        if col not in df.columns:
            df[col] = None
    df["signal"] = df["signal"].fillna("__none__")
    dupes = df[df["strategy"] != "zero_shot"].groupby(list(ARM_FIELDS) + ["seed"]).size()
    offending = dupes[dupes > 1]
    if len(offending):
        raise SystemExit(
            "duplicate (arm, seed) rows in results.csv -- refusing to guess which one is real:\n"
            f"{offending}\n"
            "Deduplicate results.csv (or drop the stale run_ids) before analysing."
        )
    return df


def protocol_of(df: pd.DataFrame, strategy: str) -> dict:
    sub = df[df["strategy"] == strategy]
    if sub.empty:
        return {}
    return {f: sub.iloc[0][f] for f in PROTOCOL_FIELDS}


def baseline_frame(df: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    sub = df[df["strategy"] == "uniform"]
    for f, v in protocol.items():
        sub = sub[sub[f] == v]
    return sub


def by_seed(df: pd.DataFrame, metric: str) -> dict:
    return dict(zip(df["seed"], df[metric]))


def compare_to_uniform(df: pd.DataFrame, strategy: str, metric: str) -> dict:
    """Compare one arm against the uniform baseline that shares its protocol.

    The MDE is computed per comparison, from a sigma pooled across *both* arms and that
    comparison's own paired n. Both details previously produced wrong verdicts: a blanket MDE
    understated the threshold for the 3-seed control arms, and a baseline-only sigma understated it
    for any arm noisier than uniform -- which, for a capacity-reallocation experiment, is the
    expected case rather than an edge case.
    """
    protocol = protocol_of(df, strategy)
    uniform_df = baseline_frame(df, protocol)
    other_df = df[df["strategy"] == strategy]

    uniform = by_seed(uniform_df, metric)
    other = by_seed(other_df, metric)
    result = paired_delta(other, uniform)

    seeds = [p["seed"] for p in result.pairs]
    a_vals = [other[s] for s in seeds]
    b_vals = [uniform[s] for s in seeds]
    deltas = [p["delta"] for p in result.pairs]

    sd_arm = pooled_sd(a_vals)
    sd_base = pooled_sd(b_vals)
    sd_pooled = pooled_two_sample_sd(a_vals, b_vals)
    mde = minimum_detectable_effect(sd_pooled, result.n) if result.n >= 2 else float("nan")
    perm = sign_flip_test(deltas) if result.n else {"p_value": float("nan"), "p_floor": float("nan"), "n": 0}

    return {
        "strategy": strategy,
        "metric": metric,
        "protocol": {k: (v.item() if hasattr(v, "item") else v) for k, v in protocol.items()},
        "mean_delta_vs_uniform": result.mean_delta,
        "n_pairs": result.n,
        "sd_uniform_arm": sd_base,
        "sd_this_arm": sd_arm,
        "sd_pooled": sd_pooled,
        "minimum_detectable_effect": mde,
        "sign_flip_p": perm["p_value"],
        "sign_flip_p_floor": perm["p_floor"],
        "pairs": result.pairs,
        "hedges_g": hedges_g(a_vals, b_vals),
        "cliffs_delta": cliffs_delta(a_vals, b_vals),
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
    """BUILD_SPEC.md 4.4's four-outcome table, extended with a fifth outcome the table doesn't name:
    every non-uniform strategy reliably *worse* than uniform. With n<=5 seeds this is a coarse
    mean-delta-vs-MDE read, not a statistical test -- a human should read the raw `comparisons`
    alongside this label, not trust it blindly. Each comparison is gated on its own MDE, computed
    from its own paired n and a sigma pooled across both of its arms.
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
    uniform_values = list(by_seed(df[df["strategy"] == "uniform"], metric).values())
    n_uniform = len(uniform_values)
    sd = pooled_sd(uniform_values) if n_uniform >= 2 else float("nan")

    comparisons = {}
    for strategy in df["strategy"].unique():
        if strategy in ("uniform", "zero_shot"):
            continue
        comparisons[strategy] = compare_to_uniform(df, strategy, metric)

    return {
        # The baseline arm's own spread. Reported as a descriptive noise floor only -- it is NOT
        # what any comparison is gated on, because it excludes the treatment arm's variance.
        "noise_floor_pooled_sd_uniform_only": sd,
        "n_uniform_seeds": n_uniform,
        "comparisons": comparisons,
        "interpretation": interpretation(comparisons, LOWER_IS_BETTER[metric])
        if not pd.isna(sd)
        else "insufficient uniform seeds for MDE",
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
