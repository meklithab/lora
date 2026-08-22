# rankalloc

At a fixed total LoRA adapter parameter count, does non-uniform layer-wise rank allocation beat
uniform allocation? `Qwen/Qwen2.5-0.5B-Instruct` on GSM8K, built for a free Kaggle/Colab T4.

See [`BUILD_SPEC.md`](BUILD_SPEC.md) for the full experiment design and the three invariants (I1
parameter matching, I2 scaling invariance, I3 compute matching) every result here is checked against,
[`CLAUDE.md`](CLAUDE.md) for the working rules, and [`NOTES.md`](NOTES.md) for the running log of
decisions, deviations, and every phase gate's result.

> ## ⚠️ Results in `results/` are stale and must be regenerated
>
> A P6 audit found a defect in the budget solver, a defect in the probe's gradient statistics, and
> a defect in the statistical gate. All three are fixed on this branch; none of the fixes can be
> applied retroactively to already-trained runs.
>
> **Do not cite any number currently in `results/`.** Concretely, in the shipped `gradnorm_prop`
> allocation, **13.6% of the total parameter budget was assigned by floating-point rounding order
> rather than by the gradient signal** — one module whose continuous target was rank 6 received
> rank 128. Across conditions the size of that artifact correlated with the reported loss penalty at
> r = 0.99, which makes it a more parsimonious explanation of the published ordering than the
> gradient signal is.
>
> Separately, the reported "exceeds MDE" verdicts used the baseline arm's variance for every
> comparison. Recomputed with a variance pooled across both arms of each comparison, **neither
> `gradnorm_prop` nor `gradnorm_inverse` clears its threshold**, and the corrected interpretation
> string reads *"allocation doesn't matter at this scale"* rather than *"uniform beats every
> non-uniform strategy"*.
>
> See [Limitations](#limitations) and [What changed in the P6 audit](#what-changed-in-the-p6-audit).
> Config schemas changed, so every `run_id` changed; a re-run will not collide with the stale rows.

## Quickstart (Kaggle)

**Turn on "Internet" in the notebook's settings panel first** -- this is the single most common
first-run failure, since every download (model, both datasets) needs it.

```bash
git clone https://github.com/meklithab/lora.git
cd lora
pip install -r requirements.txt
```

Then, in order:

```bash
python scripts/preflight.py
```

```bash
python scripts/run_single.py --smoke
```

`preflight.py` does the CUDA check, downloads, micro-batch calibration, and the tier-1 GPU-hour
projection into `results/preflight.json`. `run_single.py --smoke` runs the full pipeline on 20 steps
and 8 eval examples in under 3 minutes.

If `preflight.py` projects tier 1 over 8 GPU-hours, cut `optim.max_steps` or the train subset in
`configs/base.yaml` before running the real grid -- that's a deliberate human decision point, not
something any script decides on its own.

## Running the experiment

**A single condition**, e.g. a `gradnorm_prop` run against an already-computed probe:

```bash
python scripts/run_probe.py --set probe.task=gsm8k --set probe.rank=8 --set probe.steps=100
```

```bash
python scripts/run_single.py --condition demo --set alloc.strategy=gradnorm_prop --set alloc.probe_id=PROBE_ID
```

Every `--set key.path=value` dotted override applies on top of `configs/base.yaml` (or whatever
`--config` points at); see [`src/rankalloc/config.py`](src/rankalloc/config.py) for the full schema.
An unknown key in a YAML config or a `--set` path is a hard error by design -- a silently-defaulted
typo is how a grid produces garbage. The same strictness now applies to grid files.

**The full tier-1 grid** (17 training runs + 6 probes -- the runs that make the paper):

```bash
python scripts/run_grid.py --grid configs/grid_tier1.yaml --dry_run
```

```bash
python scripts/run_grid.py --grid configs/grid_tier1.yaml
```

Re-running the same command **resumes automatically, no flags needed**: `run_grid.py` skips any
`run_id` already present in `results/results.csv`. Kaggle/Colab sessions die without warning, so this
is the only thing that makes the grid survivable -- if a session dies mid-grid, just re-run the same
command in a fresh session after `pip install -r requirements.txt` again (packages don't persist
between Kaggle sessions, but `results/` does if it's inside your working directory / persisted
output). Add `--max_hours H` to stop launching new runs once your invocation has spent `H` GPU-hours,
useful for splitting a long grid across multiple sessions deliberately.

`--dry_run` now resolves the same `alloc.probe_id` override the real run does, so its `SKIP (done)` /
`RUN` column is accurate for the gradnorm arms too (previously it printed `run_id`s that no real run
would ever produce).

**Tier 2** (only start if >=12h remain after tier 1 -- see `BUILD_SPEC.md` §6 for why each arm exists
and the order to run them in):

```bash
python scripts/run_grid.py --grid configs/grid_tier2.yaml --dry_run
```

```bash
python scripts/run_grid.py --grid configs/grid_tier2.yaml
```

Tier 2's `gradnorm_prop` arms reuse tier 1's GSM8K probes automatically (same `probe_id`s, computed
identically from `configs/base.yaml` regardless of which grid file is running) -- you don't need to
re-run the probe.

## Analysis and figures

Both read only the `results/` tree; neither re-runs anything:

```bash
python scripts/analyze.py
```

```bash
python scripts/make_figures.py
```

`analyze.py` writes `results/analysis.json` and prints paired deltas, per-comparison noise floors,
per-comparison MDEs, effect sizes, an exact sign-flip permutation p-value reported **alongside its
own attainable floor**, and the four-outcome interpretation table (§4.4).

Two rules `analyze.py` now enforces rather than assumes:

- rows are grouped by a full **arm key** (`strategy`, `signal`, `temperature`, `scaling_mode`,
  `budget_rank`, `max_steps`), not by `strategy` alone. Tier 2 runs four `gradnorm_prop` variants at
  seeds 0-2; grouping on `strategy` alone blended them into the tier-1 arm and kept whichever row
  landed last in the file.
- duplicate `(arm, seed)` rows are a **hard error**, not silent last-write-wins. Non-experimental
  rows (`condition=smoke_test`) are dropped.

`analyze.py`'s "interpretation" string is a coarse MDE-gated read meant to sit alongside the raw
per-strategy deltas it also prints. The only inferential statistic in the pipeline is the exact
sign-flip permutation test, which is reported with its floor (`2^(1-n)`: 0.25 at n=3, 0.0625 at n=5)
so that a p pinned at the design's limit reads as *"every seed agreed and the design cannot say
more"* rather than as *"nearly significant"*.

`make_figures.py`'s six figures:

1. Allocation profiles (probe rms signal by layer, GSM8K vs Alpaca side by side). Panels built from a
   probe that predates the current fixes, or that contains non-finite entries, are **labelled
   `STALE/INCOMPLETE` and print a warning** instead of silently dropping points.
2. Forest plot (paired delta vs uniform per condition, individual seeds, **per-comparison** MDE
   intervals -- arms run at different seed counts and have different variances, so one shared band
   misrepresents every arm but the one it was computed from)
3. Learning curves (held-out loss vs cumulative GPU-seconds, by condition)
4. Noise floor (the uniform-baseline seed spread, deduplicated by seed)
5. Scaling-trap panel (`constant_ratio` vs `fixed_alpha` at identical allocations -- tier 2)
6. Budget fidelity (realised vs target parameter count per condition -- the proof of I1)

Each function skips gracefully (prints why) if its required data isn't in `results/` yet, rather than
assuming a full grid has already run.

## Method notes

### The budget solver

`solve_allocation` is a two-stage apportionment, not a proportional rounding:

1. **Water-fill.** Find `lambda` such that `sum_m c_m * clip(lambda * u_m, r_min, r_cap_m) == budget`,
   where `u_m` is the sharpened value density and `c_m = d_in + d_out`. Budget freed by a clamp is
   returned to the unclamped modules *in proportion to their own demand*. Solved by active set
   (exact division) rather than bisection, so an integral target lands on the integer exactly instead
   of at `15.999999997`.
2. **Largest remainder, one award per module.** Floor, then award `+1` in decreasing remainder order,
   each module eligible at most once.

Together these guarantee the **quota property**, asserted at runtime and reported as
`quota_max_deviation` in `results.csv`:

```
floor(rho_m) <= r_m <= ceil(rho_m)   for every module m
```

This is what makes an allocation attributable to the signal that produced it. On the real GSM8K
probe the pre-fix solver reached `max|r_m - rho_m| = 122`; the fixed solver stays below `0.54`.

`r_cap_m = min(r_max, d_in, d_out)`, because `rank(dW) <= min(d_in, d_out)` for any `dW = B @ A`.
Under GQA this binds hard: Qwen2.5-0.5B has 14 query heads and 2 KV heads, so `k_proj`/`v_proj` have
`d_out = 128`, and `r = 128` there is a *dense* reparameterisation rather than a low-rank adaptation.
The shipped `gradnorm_prop` allocation had 17 modules at or above full rank.

### Temperature

Sharpening acts on the **value density** `w_m / c_m`, never on the raw weight. Sharpening the raw
weight gives `rho_m ~ c_m^(1/T) / c_m`, which turns the *uniform baseline itself* into a width-driven
allocation at any `T != 1`. At `T = 1` the two are algebraically identical, so the tier-1 protocol is
unchanged. The density form is also what a marginal-value model implies: if the `k`-th rank unit in
module `m` is worth `v_m * k^(-T)`, equalising marginal value per unit cost gives
`r_m ~ (w_m / c_m)^(1/T)` -- cost inside the exponent.

### What the probe measures

For one micro-batch, `dL/dB = s * G @ A.T`, where `G = dL/dW` is the gradient the frozen weight would
receive. The B-gradient is therefore a **random sketch** of the frozen-weight gradient, with `A` as
the projection. With PEFT's `kaiming_uniform_` init (per-entry variance `1/(3*d_in)`):

```
rms := ||dL/dB||_F / sqrt(d_out * r) = (s / sqrt(3)) * ||G||_F / sqrt(d_in * d_out)
```

so `rms` is width-normalised in **both** dimensions and independent of the probe rank. `raw_norm`
differs from it by exactly `sqrt(d_out * r)` and is retained as the deliberate width-confound
ablation, not as a candidate signal.

The sketch identity only holds while `A` *is* the random projection, so `probe.freeze_a` (default
`true`) freezes `lora_A` for the probe's duration. Gradients are unscaled (`scaler.unscale_`) before
being read, because the dynamic loss scale changes during the probe and an average of raw per-step
norms would otherwise be a scale-weighted average.

`rms` averages per-step norms, estimating `E||G_t||` = signal + noise. The `coherent` signal instead
norms the *accumulated* gradient, `||sum_t G_t||`, estimating the consistent descent direction with
noise averaged out. Their ratio is recorded per module as `coherence` in `(0, 1]`: near 1 means every
step pointed the same way; near 0 means the module's gradient is dominated by batch noise and extra
rank there buys nothing. Both are stored so the choice stays an analysis decision.

## Tests

```bash
pytest -q
```

No GPU needed -- `test_masking.py` downloads the real `Qwen/Qwen2.5-0.5B-Instruct` tokenizer and real
GSM8K/Alpaca data but only tokenizes (no model forward pass). **447 tests as of P6**, covering
`allocation.py`'s invariants (§7, heaviest), the quota property under heavy-tailed weights, per-module
rank ceilings, temperature-invariance of the uniform baseline, the pooled-variance MDE, the exact
permutation test and its floor, the three scaling modes' `r`-exponents, response-only masking, config
strict-validation and `run_id` hashing, the GSM8K answer extractors, and `metrics.py`.

`tests/test_allocation_invariants.py` and `tests/test_stats_and_scaling.py` are the P6 regression
suite. Every test in them fails against the pre-audit code. The pre-existing allocation tests all
*passed* against the buggy solver, because they drew weights from `uniform(0.001, 1.0)` -- mild
enough that `r_max` never bound, which is exactly the regime where the defect was invisible. The new
tests use `Dirichlet(0.3)` for that reason.

If `pytest` cannot import `peft`, use the project virtualenv explicitly
(`.venv/Scripts/python -m pytest -q` on Windows); a broken system-Python `transformers` install will
otherwise fail collection on the scaling tests.

## Repository layout

```
src/rankalloc/    the package: config, seeding, data, probe, allocation, modeling, train,
                  evaluate, metrics, io_utils
scripts/          preflight, run_probe, run_single, run_grid, analyze, make_figures, dev/commit.sh
configs/          base.yaml + grid_tier1.yaml + grid_tier2.yaml
tests/            pytest suite
results/          gitignored except .gitkeep -- results.csv, runs/<run_id>/, probe/<probe_id>.json,
                  figures/, analysis.json all land here
```

## Limitations

These are ordered by how much they constrain what can be claimed. L1-L3 are open design questions
that the current protocol does not resolve; L4-L6 bound the generality of any result; L7-L8 are
artifact-hygiene issues.

### L1. The statistical unit is the allocation draw, not the training seed

The estimand is *"does the probe-to-allocation procedure beat uniform?"*, whose unit is the
**allocation draw**; training seeds are nested replicates within a draw. The published tier-1 grid
used **one** probe (seed 0), so `gradnorm_prop` and `gradnorm_inverse` each had `n = 1` at the level
of the claim regardless of how many training seeds were run — five seeds re-measured one fixed
allocation. `random`, by contrast, drew a fresh allocation per seed, so its spread includes allocation
variance while the others' does not. The two arms are therefore not comparable estimators, and
applying a single sigma to both was invalid in both directions.

`configs/grid_tier1.yaml` now sets `probe_seeds: [0, 1, 2]` and `run_grid.py` rotates each run's seed
across the available draws. This makes the allocation replicated. It does **not** make the analysis
hierarchical: `analyze.py` still treats every row as an exchangeable replicate. A proper treatment
would fit `y = mu + tau_condition + b_probe_draw + eps` (allocation draw as a random effect), or
two-stage — average within draw, then test on draw-level means. **Until that lands, read
per-condition deltas as descriptive.**

### L2. `constant_ratio` scaling probably does not isolate rank from optimisation dynamics

The design assumes that holding `alpha/r` constant makes rank the only quantity varying between
conditions. That assumption is not derivable and is likely **backwards**. With `B` initialised at
zero, AdamW drives `|B_jk|` toward `~lr*t` independently of `r`, and `||A x||` grows as `sqrt(r)`, so
the adapter's contribution scales as `s(r) * r^theta` with `theta` in `[1/2, 1]` (`1/2` if `B`'s
columns stay incoherent with `A x`, `1` if they align — which consistent gradients encourage).
Rank-neutrality therefore requires an `s`-exponent in `[-1, -1/2]`:

| mode | `s(r)` | exponent | rank-neutral? |
|---|---|---|---|
| `constant_ratio` (tier-1 default) | `alpha_ratio` | `0` | **no — outside the bracket** |
| `rslora` | `alpha_ratio * sqrt(R/r)` | `-1/2` | yes at `theta = 1/2` |
| `fixed_alpha` | `alpha_ratio * R / r` | `-1` | yes at `theta = 1` |

Under `constant_ratio`, higher-rank modules adapt **faster**, not equally. Since `gradnorm_prop`
concentrates rank, it also silently raises the effective learning rate on exactly the modules the
signal favours — the confound §3.4 claims to have eliminated, arriving through a different door.
This is consistent with the one anomalous tier-1 seed (`gradnorm_prop` seed 4 sat 0.12 nats worse at
step 50 and never recovered, despite an identical allocation and identical data to its siblings).

`theta` is an empirical property of the trajectory, not a constant, so `scaling.mode` is left as a
config choice. **The tier-2 scaling ablation is now load-bearing, not optional.** All three modes are
anchored to agree at `r = budget_rank`, so they differ only in how they extrapolate — without that
anchoring the comparison would confound "which exponent" with "how strong is the adapter overall".
A separate bug meant the `rslora` arm previously applied `s(r) = alpha_ratio * sqrt(r)`, scaling that
*grows* with rank — the opposite of the rule rsLoRA specifies. Fixed; any prior `rslora` result is
void.

### L3. Global gradient clipping couples the modules

`optim.clip_mode` defaults to `"global"`, which clips the norm over the union of all LoRA parameters.
That makes each module's effective step depend on the *other* modules' gradients, which vary with the
allocation — a second route by which capacity and step size stay entangled. `"per_module"` clips each
adapter independently and removes the coupling; `"none"` disables clipping. The default is left at
`"global"` because it is the established protocol and changing it is a research decision, not a bug
fix. `step_log.jsonl` records the global pre-clip norm in every mode, so how often the clip actually
binds is measurable — **but it has not been measured.**

### L4. The noise floor measures initialisation variance only

`data.train_order_seed` defaults to `None`, meaning every run consumes identical data in identical
order (the compute-matching invariant I3). The consequence is that the seed-to-seed spread — and
therefore every MDE derived from it — reflects **LoRA initialisation noise only**, the smallest
available noise source. The published uniform sigma of `0.00033` should be read that way. Set
`data.train_order_seed` to the run seed to fold data-order noise into the floor; expect every
threshold to widen. Results are conditional on one data ordering.

Relatedly, the design is only **nominally paired**. PEFT's `lora_A` initialisation consumes RNG per
module in an amount that depends on that module's rank, so two conditions at the same seed do not
share an initialisation — the seed indexes a replicate, not a matched pair. The independent-groups
MDE is therefore the correct form (and is what `analyze.py` uses), but no place in the design should
claim the precision that genuine pairing would buy. `paired_minimum_detectable_effect` exists in
`metrics.py` for designs that do match nuisance factors; this one does not.

### L5. The `rms` signal may be measuring architectural fan-out, not importance

`rms` is a mathematically sound per-parameter gradient magnitude (see [What the probe
measures](#what-the-probe-measures)) — but per-parameter gradient magnitude is not obviously the
quantity the research question needs. On the stored GSM8K probe, `log rms` regresses on `log d_out`
with slope `-0.43` and `70%` of its variance is explained by projection type alone. The two modules
the signal most favours (`v_proj`, `k_proj`) are exactly the two with `d_out = 128` under GQA, where
each KV output unit aggregates gradient from **7 query heads**. The predicted inflation from that
sharing alone is between `sqrt(7) ~ 2.6` and `7`; the observed `v_proj / o_proj` ratio is `3.8`.

More fundamentally, LoRA rank is a knob for *how many independent directions* an update needs, not
for how large the gradient is. A layer whose gradient is large but concentrated in one direction
needs rank 1, not rank 128. A scalar signal cannot distinguish those cases; the singular-value
spectrum of the same sketch can. The `coherent` signal and per-module `coherence` ratio are recorded
as a first step toward this, but **the allocation is still driven by a scalar, and the mechanistic
account in the paper's §5.3/§6 should be treated as unconfirmed** until the width/fan-out
decomposition is run.

### L6. Scope and effect size

Single model scale, single task, single learning rate, single sharpening setting, 400 steps. Nothing
here licenses a claim about gradient-guided allocation in general.

Effect sizes are also small in absolute terms: uniform fine-tuning moves held-out loss from `0.825`
(zero-shot) to `0.484`, a gain of `0.341` nats. The largest between-condition difference ever measured
was `0.0027` nats — **0.8% of the fine-tuning effect**. Even a fully confirmed difference of that size
would be a statement about the tail of the optimisation, not about capacity allocation mattering
much.

### L7. Held-out loss is evaluated under fp16 autocast

`compute_held_out_loss` runs under `torch.autocast(float16)` with logits upcast to fp32 only after
the forward pass. Averaged over ~30k supervised tokens the rounding error should be well below the
effect sizes involved, but this has not been verified against an fp32 re-evaluation, and the effect
sizes in question are 0.5% relative.

### L8. Artifact hygiene

- **Adapter weights were not preserved.** Every `results/runs/*/adapter/` directory holds only
  `adapter_config.json` and `README.md`. `save_pretrained` is called, so this is a transfer loss, not
  a code defect — but it blocks the single cheapest test of L2 (regress `log ||dW_m||_F` on
  `log r_m`; slope 0 means rank-neutral). Preserve `adapter_model.safetensors` on the next run.
- **`step_log.jsonl` is absent from all 18 shipped runs**, which is why the `gradnorm_prop` seed-4
  anomaly could not be diagnosed. It is written by `train.py`; preserve it.
- **The stored Alpaca probe is corrupt**, carrying `rms = NaN` for all four layer-0 attention modules
  and lacking `valid_steps` — it predates the GradScaler-overflow fix. No training run consumed it
  (`solve_allocation`'s finite-weight guard would have raised), so no result is contaminated, but
  figure 1 was built from it. `make_figures.py` now labels such panels loudly. Regenerate it.
- **Probe-seed claims need artifacts.** Only two probe JSONs exist, both seed 0. Any claim about
  probe stability across independent draws needs the corresponding files committed; `probe_seeds`
  now produces them.

## What changed in the P6 audit

| # | Defect | Where | Fix |
|---|---|---|---|
| 1 | Surplus budget funnelled into whichever module had the largest *static* fractional remainder; `max\|r-rho\|` reached 122, and 13.6% of `gradnorm_prop`'s budget was assigned by rounding order | `allocation.solve_allocation` | Water-fill + one-award-per-module largest remainder; quota property asserted at runtime |
| 2 | No per-module rank ceiling; 17 modules allocated at or above `min(d_in, d_out)` | `allocation.ModuleSpec.max_useful_rank` | `r_cap_m = min(r_max, d_in, d_out)` |
| 3 | Temperature `!= 1` deformed the *uniform baseline* into a width-driven allocation | `allocation._sharpen` | Sharpen value density `w/c`, not raw weight; identical at `T=1` |
| 4 | Probe read fp16-scaled gradients; the dynamic scale changed mid-probe | `probe.run_probe` | `scaler.unscale_` before reading `.grad` |
| 5 | `lora_A` trained during the probe, biasing the sketch identity | `probe.run_probe` | `probe.freeze_a` (default `true`) |
| 6 | Only mean-of-norms recorded (signal + noise) | `probe.run_probe` | Added `coherent` signal and per-module `coherence` ratio |
| 7 | `rslora` applied `s(r) = alpha_ratio * sqrt(r)` — scaling that grows with rank | `modeling.compute_alpha_pattern` | Constant alpha so PEFT's `/sqrt(r)` yields the canonical rule; all modes anchored at `r = budget_rank` |
| 8 | MDE used the baseline arm's sigma for every comparison; treatment arms were up to 6.3x noisier | `analyze.py`, `metrics.py` | `pooled_two_sample_sd` per comparison |
| 9 | No inferential statistic reported at all | `metrics.sign_flip_test` | Exact permutation test, reported with its attainable floor |
| 10 | Rows grouped by `strategy` alone; tier-2 variants would blend into tier-1 arms, last-write-wins | `analyze.py` | Full arm key; duplicate `(arm, seed)` is a hard error |
| 11 | `smoke_test` row inflated the plotted noise floor 96x | `make_figures.py` | Non-experimental conditions excluded; per-seed dedup enforced |
| 12 | Forest plot drew one MDE band from the last loop iteration | `make_figures.py` | Per-comparison intervals |
| 13 | `--dry_run` computed `run_id` without the `alloc.probe_id` override, so its resume column was wrong for every gradnorm arm | `run_grid.py` | Shared `with_probe_override` helper |
| 14 | Unknown keys in grid YAML silently ignored | `run_grid.load_grid` | Hard error, matching `config.py` |

New config fields: `data.train_order_seed`, `probe.freeze_a`, `optim.clip_mode`. New `results.csv`
columns: `quota_max_deviation`, `clip_mode`. New grid key: `probe_seeds`.

## Definition of done (BUILD_SPEC.md §8)

- `pytest` green -- `pytest -q` (447 tests)
- Smoke run under 3 minutes -- `time python scripts/run_single.py --smoke` (needs a real GPU to
  actually meet the 3-minute bar; verified 1m50s on a Tesla T4)
- Resume survives `SIGKILL` -- kill `run_grid.py` mid-grid, re-run the same command, check
  `results/results.csv` for duplicate or missing `run_id`s (there should be none)
- Every condition's `budget_rel_error` < 0.5% **and** `quota_max_deviation` < 1.0 -- both are columns
  in `results/results.csv`
- `adapter_params_verified` matches `alloc.json` for all runs -- checked automatically inside
  `run_single.py` itself (an `assert` before training starts, not a post-hoc check)
- All six figures regenerate from the `results/` tree alone -- `python scripts/make_figures.py`
- `adapter_model.safetensors` and `step_log.jsonl` present for every run (see L8)
