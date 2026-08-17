# `rankalloc` — Build Specification
**Read this whole file before writing any code. Then enter plan mode and produce a phase plan for my
approval. Do not start implementing until I approve the plan.**
This is a research codebase with a 40-hour deadline. Correctness of the *comparison* matters more
than features. A fast implementation that silently breaks an invariant in §1 is worth less than
nothing, because it produces numbers that look publishable and aren't.
---
## 0. Working agreement
- **Phase gates.** Work through §8 in order. At the end of each phase: run the tests, commit, and
  **STOP and report to me** with what passed, what you changed from spec, and what you're unsure
  about. Do not roll two phases into one commit.
- **Never edit a test to make it pass.** If a test in §7 fails, either the implementation is wrong or
  the spec is wrong. Fix the implementation, or stop and tell me the spec is wrong. Deleting,
  loosening, or `xfail`-ing a specified test is the one thing that ends the run.
- **No silent substitutions.** If a library API differs from what this spec assumes, stop and say so
  rather than inventing an equivalent.
- **Keep `NOTES.md`** at the repo root: a running log of decisions, surprises, deviations, and
  timings. Append to it every phase. This is context that survives compaction — treat it as
  load-bearing.
- **Ask before scope changes.** If something here looks like it will take more than ~2 hours, stop
  and flag it instead of building it.
---
## 1. The three invariants
The whole experiment asks: **at a fixed total LoRA adapter parameter count, does non-uniform
layer-wise rank allocation beat uniform allocation?** Three properties must be guaranteed *by code*,
because the paper cannot defend them otherwise. Every one gets an assertion and a test.
**I1 — Parameter matching.** Every condition in a comparison spends the same adapter parameter
budget, within a logged tolerance. Cost of a LoRA adapter on a linear layer is `r · (d_in + d_out)`.
Matching the *sum of ranks* is **not** matching parameters: Qwen2.5-0.5B uses grouped-query
attention, so `k_proj` is 896→128 while `gate_proj` is 896→4864. Match `Σ r_m · c_m` where
`c_m = d_in + d_out`.
**I2 — Scaling invariance.** LoRA scales its update by `α/r`. Vary rank per layer with global `α`
fixed and the per-layer *effective step size* changes too — the experiment then measures capacity and
learning rate mixed together. The default holds scaling constant across layers (§4.4).
**I3 — Compute matching.** Every condition trains for the **same number of optimizer steps** on the
same data in the same order. Not the same number of epochs, not "until converged". A fixed step
budget makes the compute comparison exact and removes a whole class of reviewer objections.
---
## 2. Environment
Target: **Kaggle / Colab free tier.** Build to these, not around them.
- Single **T4 (16 GB)** or **P100**. **fp16 only — T4 has no bf16.** Emit no bf16 autocast paths.
- No FlashAttention-2 (Ampere+ only). Use `attn_implementation="sdpa"`.
- Sessions die without warning (12 h cap on Kaggle, less on Colab). Everything long-running must
  resume from disk with no flags and no manual bookkeeping.
- **Kaggle requires "Internet on" in notebook settings** for HuggingFace downloads. Put this in the
  README quickstart — it is the single most common first-run failure.
- Local JSONL + CSV logging only. No W&B, no `trl`, no `bitsandbytes` (the last avoids a family of T4
  failures and we don't need quantisation).
Pin in `requirements.txt`: `torch`, `transformers`, `peft`, `datasets`, `accelerate`, `numpy`,
`pandas`, `scipy`, `matplotlib`, `pyyaml`, `pytest`.
---
## 3. Repository layout
```
rankalloc/
├── README.md              # quickstart, grid, resume instructions, figure regeneration
├── CLAUDE.md              # invariants + working rules (provided separately — do not overwrite)
├── NOTES.md               # your running log
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── grid_tier1.yaml    # the runs that make the paper
│   └── grid_tier2.yaml    # depth, only if time remains
├── src/rankalloc/
│   ├── config.py          # dataclasses, YAML, CLI override, run_id hashing
│   ├── seeding.py
│   ├── data.py             # load, format, response-only masking, splits, token accounting
│   ├── probe.py            # gradient-sensitivity probe
│   ├── allocation.py       # ★ budget solver + strategies — the core contribution
│   ├── modeling.py         # model + LoRA construction, scaling modes, param verification
│   ├── train.py            # fixed-step training loop
│   ├── evaluate.py         # held-out loss + GSM8K generation eval
│   ├── metrics.py          # paired deltas, MDE, effect sizes
│   └── io_utils.py         # atomic CSV append, run dirs, manifest
├── scripts/
│   ├── preflight.py        # ★ Phase 0: GPU check, model download, batch calibration, time projection
│   ├── run_probe.py
│   ├── run_single.py
│   ├── run_grid.py         # resumable driver
│   ├── analyze.py
│   ├── make_figures.py
│   └── dev/commit.sh       # backdated commit helper (§9)
├── tests/
│   ├── test_allocation.py  # heaviest
│   ├── test_masking.py
│   ├── test_config.py
│   └── test_determinism.py
└── results/                # gitignored except .gitkeep
    ├── results.csv
    └── runs/<run_id>/
```
---
## 4. Module specifications
### 4.1 `config.py`
Frozen dataclasses composed into `RunConfig`. Requirements:
- YAML → dataclass with **strict key validation**: unknown key is a hard error. A typo'd
  hyperparameter that silently defaults is how a grid produces garbage.
- CLI override, dotted: `--set optim.lr=1e-4 --set alloc.temperature=0.5`, coerced from annotations.
- `run_id` = first 12 hex of `sha256(canonical_json(resolved_config, sort_keys=True))`, **including
  seed**. This is the resumability primitive.
- `to_json()` round-trips exactly.
### 4.2 `data.py`
Primary task **GSM8K** (`openai/gsm8k`, config `main`); second task **Alpaca** (`tatsu-lab/alpaca`)
used for the probe only in tier 1. Interface: `load_task(cfg) -> TaskBundle` with `train`,
`val_loss`, `val_gen`, `stats`.
- **Deterministic splits** from a fixed `split_seed` independent of the run seed, so every condition
  sees identical data in identical order. Assert three-way index disjointness.
- **Response-only loss masking**: prompt tokens get label `-100`. This is the most commonly botched
  step in the whole pipeline — see the test in §7.
- Chat template when the tokenizer provides one, with a hard-coded fallback recorded in `stats` so
  the paper can quote the exact format.
- **Token accounting**: total and supervised token counts go in `results.csv`. Under I3 these must be
  *identical* across conditions — assert it and fail loudly, since any difference means a pipeline
  bug.
- `val_loss` = 300 held-out examples. `val_gen` = 200 fixed GSM8K test indices.
- Truncate at `max_seq_len` (512), log the truncated count.
### 4.3 `probe.py`
`run_probe(cfg) -> ProbeResult`. Build the model with **uniform** rank `probe_rank` (8), train
`probe_steps` (100) at the real LR and warm-up schedule, accumulating per-module gradient statistics
on **`lora_B`** — B is zero-initialised, so its gradient carries signal from step 0 while A's is zero
at init.
Accumulate all four signals so allocation can be recomputed without re-probing:
| Key | Definition | Role |
|---|---|---|
| `rms` | mean over steps of `‖g‖₂ / √numel` | **default** — width-normalised |
| `raw_norm` | mean of `‖g‖₂` | deliberately-flawed comparison arm |
| `fisher` | mean of `g²`, summed, `/numel` | curvature-flavoured variant |
| `relative` | mean of `‖g‖₂ / (‖W‖₂ + ε)` | relative update magnitude |
> **Do not quietly make `raw_norm` the default because it is simpler.** Raw gradient norm scales with
> module width, so wide MLP projections dominate for reasons unrelated to importance. It exists only
> so the paper can show what the naive choice would have concluded.
Record `probe_wall_seconds` and `probe_gpu_seconds` — the probe is part of the method's cost and must
be reportable as such. Output to `results/probe/<probe_id>.json` with per-module metadata
(`in_features`, `out_features`, `numel`, `layer_idx`, `proj_type`); `probe_id` hashes the probe config
so a stale probe can never be silently reused.
**Run the probe on both GSM8K and Alpaca.** Comparing the two allocation profiles is a tier-1
deliverable (Figure 1) and costs only probe time, not training time.
### 4.4 `allocation.py` — the core
Give this the most care and the most tests.
**Budget.** `c_m = d_in + d_out`; `params(m, r) = r · c_m`; budget `B = R · Σ_m c_m` for reference
uniform rank `R`.
**Problem.** Given non-negative weights `w_m` (normalised), spend the *parameter* budget in
proportion to weight, then convert to integer rank:
`ideal_params_m = B · w_m`, so `ideal_rank_m = B · w_m / c_m`.
```python
def solve_allocation(modules, weights, budget, r_min=1, r_max=128, temperature=1.0) -> Allocation
```
Algorithm — implement exactly, it is testable:
1. **Sharpen**: `w_m ← w_m ** (1/temperature)`, renormalise. `temperature→0` concentrates,
   `→∞` approaches uniform. An ablation axis, not decoration.
2. **Floor**: `r_m = clamp(floor(B · w_m / c_m), r_min, r_max)`.
3. **Spend up**: while a legal +1 exists, give it to the module with the largest fractional remainder
   among those with `r_m < r_max` and `c_m ≤ B − spent`. Tie-break on module name so the solver is
   seed-independent.
4. **Spend down**: while `spent > B`, take −1 from the smallest remainder among `r_m > r_min`.
5. Stop when no legal move reduces `|spent − B|`.
**Asserted invariants:** `|spent − B| ≤ max_m c_m` (the integrality floor); rank clamps respected;
uniform weights with integral `R` reproduce `r_m = R` exactly with zero error; deterministic;
invariant to module-list permutation.
**Strategies** — `gradnorm_inverse` and `random` are not optional; they are what makes a positive
result believable and a null result interpretable.
| Name | Weight rule | Role |
|---|---|---|
| `uniform` | `w_m ∝ c_m` ⇒ equal rank | baseline |
| `gradnorm_prop` | `w_m ∝ signal_m` | hypothesis |
| `gradnorm_inverse` | `w_m ∝ 1/(signal_m + ε)` | **directional control** |
| `random` | `w_m ∝ Dirichlet(α=1)`, seeded by run seed | **non-uniformity control** |
| `early_heavy` | `w_m ∝ exp(−λ·layer_idx/L)` | tier 2 |
| `late_heavy` | `w_m ∝ exp(+λ·layer_idx/L)` | tier 2 |
**Interpretation table — put this in `analyze.py`'s output and in the paper.** The two controls
disambiguate four distinct worlds, and this is the analysis that makes the paper:
| Outcome | Conclusion |
|---|---|
| prop ≈ inverse ≈ random ≈ uniform | allocation doesn't matter at this scale; report the MDE bound |
| prop ≈ inverse ≈ random > uniform | non-uniformity itself helps; the gradient signal is irrelevant |
| prop ≈ inverse > random | the signal identifies *something*, but direction doesn't matter — suspect a width or position correlate |
| prop > random > inverse | the gradient signal is informative **and directional** — the strong result |
`Allocation` → `runs/<run_id>/alloc.json`: `rank_pattern`, `alpha_pattern`, realised `params_total`,
`budget`, `abs_error`, `rel_error`, per-module table, strategy, temperature, probe id, weight vector.
### 4.5 `modeling.py` — the scaling trap (I2)
```python
scaling_mode: Literal["constant_ratio", "rslora", "fixed_alpha"] = "constant_ratio"
```
- **`constant_ratio`** (default): `α_m = k · r_m` with `k = alpha_ratio` (2), so `α_m/r_m = k` for
  every module. Capacity varies; effective step size does not.
- **`rslora`**: PEFT's `use_rslora=True`, giving `α/√r_m` — the rank-stabilised alternative. A
  different, also-defensible normalisation. Tier-2 ablation arm.
- **`fixed_alpha`**: the naive `α_m = α` everywhere. Retained **explicitly as an ablation** so the
  paper can quantify how much the naive implementation would have misled. That comparison is cheap
  and is a genuine finding.
**Scaling mode must be identical across all conditions within a comparison.** Never vary it per
condition inside a single table.
After wrapping, **verify against the live model**: walk `named_modules()`, read back each adapter's
actual `r` and `lora_alpha`, recompute parameter count from tensor shapes, and assert it matches
`alloc.json`. Trust the model, not the config dict. Write `adapter_params_verified` to
`metrics.json`.
Base model `Qwen/Qwen2.5-0.5B-Instruct`, `torch_dtype=float16`, `attn_implementation="sdpa"`,
gradient checkpointing on, `use_cache=False` while training. Targets:
`["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`.
### 4.6 `train.py`
Hand-written loop (no `Trainer`) so timing and logging are unambiguous.
- **Fixed optimizer-step budget** (`optim.max_steps`, default 400) — not epochs. This is I3.
- AdamW, cosine schedule with warm-up ratio, `max_grad_norm=1.0`, fp16 `GradScaler`.
- Micro-batch and gradient accumulation are **fixed for the entire grid** by `preflight.py`, never
  adapted at runtime. If a run OOMs, fail it loudly and record `status=oom`; do not silently retry at
  a different batch size, which would change numerics between conditions.
- `step_log.jsonl`: step, loss, lr, grad_norm, tokens_seen, cumulative gpu_seconds.
- Timing via `torch.cuda.Event` (GPU time) **and** wall clock, reported separately.
- Periodic held-out loss for learning curves — these are a result, not a diagnostic.
- Save adapter weights only.
- `--smoke`: 20 steps, 8 eval examples, must finish under 3 minutes and touch every code path.
### 4.7 `evaluate.py`
**Primary — held-out loss.** Deterministic, low variance, cheap, and not circular here (no condition
selects data by loss). Report both `loss_token_weighted` (total NLL ÷ supervised tokens) and
`loss_example_mean`; they diverge under length imbalance, so state which the paper uses.
**Secondary — GSM8K exact match.** Greedy, `max_new_tokens=256`, left padding, batched. Two
extractors both reported: `strict` (number after final `####`) and `flexible` (last number in the
generation). Normalise commas, currency symbols, trailing zeros; compare as `Fraction` where
parseable. Log 30 generations per run to `samples.jsonl` for qualitative error analysis.
Generation eval is the expensive half of a run — gate behind `eval.run_generation` so it can be
deferred to a second pass if the clock bites.
**Also evaluate the untuned model once** as a zero-shot reference row. Without it there is no
evidence that training moved anything at all.
### 4.8 `metrics.py`
**No significance tests.** At n=3 the Wilcoxon signed-rank floor is p=0.25 and at n=5 it is p=0.0625
— the test *cannot* return significance regardless of effect size, so reporting it is misleading.
Report instead:
- `paired_delta(a, b, by="seed")` — mean paired difference plus **every individual seed pair**, which
  gets plotted.
- `pooled_sd(...)` — the noise floor from the 5 uniform-baseline seeds.
- **`minimum_detectable_effect(sigma, n, alpha=0.05, power=0.8)` = `(t_{1−α/2,df} + t_{power,df}) · σ ·
  √(2/n)`.** Put this in the abstract. It converts "we found nothing" into "we exclude effects larger
  than X", which is a real claim.
- `hedges_g`, `cliffs_delta` — effect sizes.
---
## 5. Runner and outputs
`run_grid.py`: expands the grid config, loads `results.csv`, **skips completed `run_id`s with no
flags**, runs sequentially, appends each row immediately via atomic write (temp + `os.replace`).
On per-run exception: traceback to `runs/<run_id>/error.log`, row marked `status=failed`, continue.
`--dry_run` prints the run list and projected GPU-hours using the calibration constant from
preflight. `--max_hours H` stops launching once the projection exceeds the session budget.
`results.csv`: `run_id, condition, seed, strategy, signal, temperature, scaling_mode, budget_rank,
adapter_params_verified, budget_abs_error, budget_rel_error, train_tokens, supervised_tokens,
max_steps, loss_token_weighted, loss_example_mean, gsm8k_strict, gsm8k_flexible, train_gpu_seconds,
train_wall_seconds, samples_per_sec, eval_gpu_seconds, probe_gpu_seconds, peak_vram_mb, gpu_name,
status, git_sha, timestamp`.
`make_figures.py` → six figures, each regenerable from `results.csv` alone:
1. **Allocation profiles** — rank vs layer index per strategy, **GSM8K vs Alpaca side by side.**
   Publishable even if every performance delta is null: it shows where the signal says capacity
   belongs, and whether that differs by task type.
2. **Forest plot** — paired delta vs `uniform` per condition, individual seed points overlaid, MDE
   band shaded.
3. **Learning curves** — held-out loss vs GPU-seconds, by condition. Convergence differences can
   exist where endpoint differences don't.
4. **Noise floor** — the 5-seed uniform spread every other claim is judged against.
5. **Scaling-trap panel** — `constant_ratio` vs `fixed_alpha` at identical allocations.
6. **Budget fidelity** — realised vs target parameter count per condition; the proof of I1.
---
## 6. Experiment grid
**Tier 1 — 16 training runs + 1 eval-only + 2 probes. This is the paper. Build for this.**
```yaml
budget_rank: 16
scaling_mode: constant_ratio
signal: rms
max_steps: 400
conditions:
  - {strategy: zero_shot,        n_seeds: 1, train: false}
  - {strategy: uniform,          n_seeds: 5}    # noise floor
  - {strategy: gradnorm_prop,    n_seeds: 5}    # hypothesis
  - {strategy: gradnorm_inverse, n_seeds: 3}    # directional control
  - {strategy: random,           n_seeds: 3}    # non-uniformity control
probes: [gsm8k, alpaca]
```
**Tier 2 — only start an arm if ≥12 h remain. Each is independent; do them in this order:**
```yaml
- {strategy: gradnorm_prop, scaling_mode: fixed_alpha, n_seeds: 3}   # the scaling-trap result
- {strategy: gradnorm_prop, signal: raw_norm,          n_seeds: 3}   # width-confound result
- {strategy: early_heavy,   n_seeds: 3}
- {strategy: late_heavy,    n_seeds: 3}
- {strategy: gradnorm_prop, scaling_mode: rslora,      n_seeds: 3}
- {strategy: gradnorm_prop, temperature: [0.5, 2.0],   n_seeds: 3}
- adalora_reference: {n_seeds: 3}   # ← see caveat
```
> **AdaLoRA caveat.** PEFT ships `AdaLoraConfig`, but AdaLoRA starts at `init_r` above its target and
> prunes down, so its parameter budget is *transient* and cannot be cleanly matched to ours.
> Include it only as an explicitly **non-parameter-matched reference arm**, reported with its own
> throughput (`samples_per_sec`) and peak-parameter numbers, and label it as such in every table.
> Never place it inside a parameter-matched comparison. If this looks like more than an hour of work,
> skip it — it is the most droppable item in the spec.
---
## 7. Tests — all green before any grid run
`tests/test_allocation.py` (heaviest):
- uniform weights + integral `R` ⇒ exactly `r_m = R`, zero budget error
- property test: budget error ≤ `max_m c_m` across 200 random weight vectors
- clamps respected under extreme weights (one module at 0.999)
- permutation invariance of the module list
- determinism across repeated calls
- `temperature→∞` approaches uniform; `→0` concentrates
- parameter count recomputed from `rank_pattern` equals the solver's `spent`
`tests/test_masking.py`: on ≥20 real tokenized examples, non-masked label count equals tokenized
response length, and the first non-masked position is exactly the first response token.
`tests/test_config.py`: unknown YAML key raises; `run_id` stable under dict reordering, changes when
any value changes; `to_json()` round-trips.
`tests/test_determinism.py`: same `run_id` twice at smoke scale agrees on held-out loss to
`atol=5e-3`. fp16 reductions leave residual nondeterminism — document this, don't chase bitwise.
---
## 8. Phases
Commit and **stop for review** at each gate.
- **P0 — Preflight.** `preflight.py`: assert CUDA, print GPU name and VRAM, download model and both
  datasets, calibrate the largest micro-batch that fits at `max_seq_len=512`, time 20 steps, and
  **print projected GPU-hours for tier 1**. *Gate: report the projection to me before proceeding —
  if tier 1 projects over 8 GPU-hours we cut `max_steps` or the train subset, and I decide that, not
  you.*
- **P1 — Config, seeding, io_utils, data.** Gate: `test_config.py` + `test_masking.py` green.
- **P2 — `allocation.py` + full test file.** No model code yet; this is pure and fully testable.
  Gate: `test_allocation.py` green, every invariant asserted.
- **P3 — `modeling.py` + `probe.py`.** Gate: live-model parameter verification passes for uniform and
  for a skewed allocation; both probes run and write JSON.
- **P4 — `train.py` + `evaluate.py`.** Gate: `run_single.py --smoke` completes under 3 minutes and
  writes a full run directory.
- **P5 — `run_grid.py`, `analyze.py`, `make_figures.py`.** Gate: `--dry_run` lists tier 1 correctly;
  resume verified by killing mid-run (`SIGKILL`) and restarting with no duplicated or lost rows.
- **P6 — README.** Exact commands to reproduce every number and figure.
**Definition of done:** `pytest` green; smoke run under 3 min; resume survives `SIGKILL`; every
condition's `budget_rel_error` < 0.5% and in a table; `adapter_params_verified` matches `alloc.json`
for all runs; all six figures regenerate from `results.csv` alone.
---
## 9. Git and repository conventions
**Identity — do this before the first commit.**
```bash
git config user.name  "<MY NAME>"      # ask me for these values at P0; do not guess
git config user.email "<MY EMAIL>"
```
All commits are authored as me. **Do not add `Co-Authored-By:` trailers, do not add "Generated with
Claude Code" or any similar attribution line, and do not mention Claude, AI assistance, or this spec
in any commit message or in the repository.** Commit messages describe the change, nothing else.
**Clear the existing project — destructive, gated on my confirmation.** This repository currently
contains an unrelated codebase and its commit history. Both go. **Ask me to confirm I have a backup
or that the old project is pushed elsewhere before running any of this** — the history is not
recoverable afterwards.
```bash
git checkout --orphan fresh-start     # new branch with no parent commits
git rm -rf .                          # clear the index and working tree
# ... scaffold the new project here, then make the first commit via commit.sh ...
git branch -D main
git branch -m main
git push --force origin main
```
Verify with `git log --oneline` that exactly the new commits are present, and confirm the old
branches and tags are gone (`git branch -a`, `git tag -l`) before pushing.
**Repository rename.** Rename it to `rankalloc`:
```bash
gh repo rename rankalloc                                   # if gh is authenticated
git remote set-url origin git@github.com:<USER>/rankalloc.git   # otherwise, after renaming in the web UI
```
**Branches.** `main` stays green — never commit a failing test suite to it.
- One branch per phase: `feat/p1-config-data`, `feat/p2-allocation`, `feat/p3-modeling-probe`,
  `feat/p4-train-eval`, `feat/p5-runner-analysis`.
- Merge with `git merge --no-ff` so the phase structure stays visible in the history.
- Experiment runs and results go on `exp/tier1` and `exp/tier2`, not on `main`.
- Bugfixes found during experiments: `fix/<short-description>`, merged back to `main`.
**Commit dates.** Author and committer dates run from **August 17** — the day the repository was
created — through today, spread across four days:
| Phase | Date |
|---|---|
| P0 preflight + scaffold | 2026-08-17 |
| P1 config, data | 2026-08-17 |
| P2 allocation + tests | 2026-08-18 |
| P3 modeling, probe | 2026-08-18 |
| P4 train, eval | 2026-08-19 |
| P5 runner, analysis, figures | 2026-08-20 |
| P6 README | 2026-08-20 |
Implement `scripts/dev/commit.sh`:
```bash
#!/usr/bin/env bash
# usage: scripts/dev/commit.sh "2026-08-17T14:20:00" "commit message"
set -euo pipefail
export GIT_AUTHOR_DATE="$1"
export GIT_COMMITTER_DATE="$1"
git commit -m "$2"
```
Vary the times within each day so a phase's commits are spread across working hours rather than
sharing a timestamp. Use this script for every commit.
---
## 10. Non-goals
Multi-GPU. Distributed training. Quantisation. Full fine-tuning arms. Per-condition hyperparameter
search — **LR is fixed across conditions by design**, and tuning it per condition would destroy the
comparison. Any task beyond the two behind `load_task`. A web UI. A CLI beyond the listed scripts.
