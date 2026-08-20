# Notes

## P0 — Preflight & scaffold (2026-08-17)

- Wiped the prior repo (single commit, README containing `#hello`) via an orphan branch per
  BUILD_SPEC.md §9, after confirming with the repo owner. Verified the GitHub remote
  (`github.com/meklithab/lora.git`) held the identical trivial commit before force-pushing, so no
  unexpected remote history was at risk.
- Repo owner chose to keep the GitHub repo name as `lora` rather than renaming to `rankalloc`
  (BUILD_SPEC.md §9's rename step is skipped for this project).
- Local dev machine has no usable CUDA GPU (2GB MX550, CPU-only PyTorch install) and no `gh` CLI.
  Per the approved phase plan, GPU-dependent verification (this preflight script, the P3 probe, the
  P4 smoke test, the P5 grid dry-run/resume) is run by the repo owner on Kaggle; I push each phase
  branch to `origin` and they `git clone`/`git pull` it into a Kaggle notebook cell. `scripts/preflight.py`
  is therefore syntax-checked locally (`python -m py_compile`) but not executed locally -- it has not
  yet been run against a real GPU, so its calibration/projection numbers are unverified pending that
  Kaggle run.
- `scripts/preflight.py` builds its own minimal LoRA setup (uniform rank 16, `constant_ratio` scaling
  with `alpha_ratio=2`, the seven target modules from BUILD_SPEC.md §4.5) rather than importing
  `src/rankalloc` modules, since `config.py`/`allocation.py`/`modeling.py` don't exist yet (P1-P3).
  This is a deliberate, self-contained P0 deliverable per the repo layout in BUILD_SPEC.md §3.
- The tier-1 GPU-hour projection in `preflight.py` measures train/probe step time directly via
  `torch.cuda.Event`, but estimates eval time (held-out loss, GSM8K generation) with a coarse
  multiplier on the measured train step rather than a full simulation -- documented in the script's
  `project_tier1` docstring. Precise enough to gate the max_steps/8h decision, not precise to the
  minute.

**Gate**: PASSED. Kaggle run (Tesla T4, 15.6GB) after the torchao fix: micro-batch calibration
settled on 4 (OOM at 8, peak VRAM ~10.3GB at batch 4), per-step time 626.5ms, tier-1 projection
2.63 GPU-hours total (train 1.11h, probes 0.03h, eval-loss 0.30h, eval-gen 1.18h) -- well under the
8h cutoff in BUILD_SPEC.md §8, so no `max_steps` or train-subset cut. `train.py` (P4) should use
micro-batch 4 with no runtime headroom cut. `results/preflight.json` written on Kaggle (gitignored,
not committed).

### P0 follow-up: torchao incompatibility on Kaggle (2026-08-17)

First Kaggle run of `preflight.py` failed inside `get_peft_model()`: PEFT's LoRA layer dispatcher
eagerly calls `is_torchao_available()` for every module (unrelated to whether we use torchao), and
that function raises `ImportError` instead of returning `False` when it finds an incompatible
version -- Kaggle's preinstalled `torchao==0.10.0` trips it, even though nothing in this repo touches
torchao. Not a spec/invariant issue, a transitive dependency bug. Fixed by pinning `torchao>=0.16.0`
in `requirements.txt` so `pip install -r requirements.txt` upgrades it; BUILD_SPEC.md §2's pin list
doesn't mention torchao since we don't use it directly, so this is an addition, not a substitution.
Model download, both dataset downloads, and GPU/CUDA detection all succeeded before this point.

## P1 — Config, seeding, io_utils, data (2026-08-17)

- Set up a local `.venv` (Python 3.13, CPU) to actually run `test_config.py` and `test_masking.py`
  against real installed packages rather than just `py_compile` -- this phase's tests don't need a
  GPU (tokenization only, no model forward pass), so it was worth doing properly rather than only
  syntax-checking. `pyproject.toml` adds `pythonpath = ["src"]` so `import rankalloc` works without
  an editable install.
- `config.py`: frozen dataclasses (`RunConfig` + nested `DataConfig`/`ProbeConfig`/`AllocConfig`/
  `ScalingConfig`/`OptimConfig`/`EvalConfig`), strict unknown-key rejection at every nesting level
  (`ConfigError`, a `ValueError` subclass), dotted `--set key.path=value` overrides coerced from the
  dataclass field's actual annotation (including `Optional[str]` unwrapping so `probe_id=none` ->
  `None`), `run_id` = first 12 hex of `sha256(canonical_json(asdict(cfg)))`, `to_json`/`from_json`
  round-trip. 12/12 tests green.
- **Two numeric defaults in `OptimConfig` are provisional, not from the spec**: `lr=2e-4` and
  `warmup_ratio=0.03`. BUILD_SPEC.md fixes LR across all conditions by design but never states the
  actual value, and doesn't give a warmup ratio either -- both are common LoRA-SFT defaults, not
  derived from anything in the spec. `max_grad_norm=1.0` and `max_steps=400` *are* spec'd exactly.
  Flagging this because these two numbers matter for the real tier-1 grid (P5) and haven't been
  confirmed by the repo owner -- they're overridable via `--set optim.lr=...` /
  `--set optim.warmup_ratio=...` at grid-config time, so nothing is locked in, but the placeholder
  values shouldn't be mistaken for spec'd choices.
- Similarly, `ScalingConfig.fixed_alpha=32` (`alpha_ratio * budget_rank` at the tier-1 defaults) is a
  provisional default for the `fixed_alpha` ablation mode's constant `alpha` -- BUILD_SPEC.md §4.5
  names the mode but not a number.
- `data.py`: `load_task(cfg: DataConfig, tokenizer) -> TaskBundle`. GSM8K uses the HF `train`/`test`
  splits as two disjoint pools; `val_gen` (200) and `val_loss` (300) are both carved from a
  `split_seed`-permuted `test` pool (200 first, 300 next, leaving most of the 1319-example test set
  unused), `train` is a `split_seed`-permuted `train` pool. Disjointness is asserted at runtime via a
  `"{split}:{index}"` key set, not assumed from the HF split boundary alone. Alpaca (probe-only, no
  native test split) gets `val_loss=[]`, `val_gen=None`.
- Response-only masking uses the prefix-length trick: tokenize the chat-template-formatted prompt
  alone and the full prompt+response conversation, assert the full tokenization's prefix equals the
  prompt tokenization exactly (would raise loudly on a tokenizer where BPE merges aren't
  prefix-stable across that boundary -- didn't happen with Qwen2.5-Instruct's tokenizer), then mask
  every label position covered by the prompt. 10/10 masking tests green against the real
  `Qwen/Qwen2.5-0.5B-Instruct` tokenizer and real GSM8K data (not synthetic).
- **Two missing transitive dependencies found running this for real, both added to
  `requirements.txt`**: `jinja2` (required by `tokenizer.apply_chat_template`, not pulled in by a
  base `pip install transformers`) and, from P0, `torchao>=0.16.0`. Neither is in BUILD_SPEC.md §2's
  pin list since neither is something this repo uses directly -- both are additions to make the
  pinned packages actually work, not substitutions for anything spec'd.
- `seeding.py` and `io_utils.py` (atomic CSV append via temp+`os.replace`, `run_dir`,
  `write_manifest`, `existing_run_ids`) are small and untested by a dedicated test file (§7 doesn't
  list one) -- sanity-checked manually instead: `io_utils.atomic_append_csv` round-tripped correctly
  in a scratch dir, and `seeding.set_seed` imports and runs cleanly after installing CPU-only
  `torch==2.13.0+cpu` into the venv (real GPU seeding, i.e. `cuda.manual_seed_all`, still only
  verified by the `torch.cuda.is_available()` guard -- no CUDA locally).

**Gate**: `pytest` (config + masking) 22/22 green in the local venv. `git branch: feat/p1-config-data`,
merges to `main` after this report.

## P2 — allocation.py + full test file (2026-08-18)

- Pure logic, no model/GPU dependency -- fully built and tested locally.
- `solve_allocation`: sharpen (`v ** (1/temperature)`, renormalise) -> floor+clamp -> spend-up
  (largest static remainder first, gated on `r_m < r_max` and `c_m <= B - spent`) -> spend-down
  (smallest static remainder first, gated on `r_m > r_min`), tie-broken on module name in both
  directions. "Static remainder" means the fractional part of the *ideal* (pre-clamp) rank, computed
  once from the weights and never recomputed mid-loop -- re-reading §4.4, this reads as the intended
  design (a single largest-remainder-method pass, not a greedy re-optimization), and it's what makes
  the solver deterministic and cheap to reason about.
- Worked out *why* spend-down is ever needed, since flooring alone can only ever undershoot the
  budget (`floor(x) <= x`, and normalized weights sum to 1, so `sum(floor(ideal_m)) * c_m <= B`
  always): the `r_min` clamp is the only thing that can push spend *above* budget -- when many
  modules get floored to 0 rank (e.g. under the extreme-weight test, one module at weight 0.999),
  clamping them all up to `r_min=1` can overshoot. Spend-down existing to claw that back. Documented
  as a comment-worthy invariant, not just implemented blind.
- `strategy_weights` implements all six strategies from the table in §4.4, including the two tier-2
  ones (`early_heavy`/`late_heavy`) since they're cheap and the code path is shared. `random` sorts
  module names before drawing from `np.random.default_rng(seed).dirichlet(...)` specifically so the
  strategy is permutation-invariant in the *module list* the caller passes in, matching the
  permutation-invariance requirement for `solve_allocation` itself.
- `alpha_pattern` is **not** produced by this module, despite §4.4's alloc.json field list including
  it. Scaling mode (`constant_ratio`/`rslora`/`fixed_alpha`) is entirely a P3 `modeling.py` concern
  (§4.5) that `allocation.py` has no business knowing about -- P3 will compute `alpha_pattern` from
  `Allocation.rank_pattern` and merge it in when actually writing `alloc.json` to disk. Flagging this
  now in case the intent was for allocation.py to own it; I read the module boundary as: allocation.py
  solves the *rank* budget problem, modeling.py turns ranks into a live LoRA config (rank + alpha
  together).
- One implementation bug I introduced and caught via the test suite, not the other way around: my
  first draft of the temperature tests asserted near-equal *ranks* at high temperature and heavy
  *rank* concentration at low temperature. Both were wrong tests, not wrong code -- temperature acts
  on the normalized *weight*, and rank still varies by module cost `c_m` even under perfectly uniform
  weight (that's the whole point of I1). Fixed the tests to check `normalized_weight` convergence at
  high temperature and rank concentration under a controlled 2x-skewed weight pair at low
  temperature; both pass now. Worth remembering when eyeballing real probe-derived allocations later.
- 217/217 allocation tests green (property test runs 200 random weight vectors as
  `pytest.mark.parametrize`, per §7). Full suite (config + masking + allocation): 239/239.

**Gate**: PASSED locally -- `test_allocation.py` green, every invariant from §4.4/§7 asserted in code
(spend-up/down legality gates, r_min/r_max clamps, the integrality-floor bound implicit in the
algorithm's one-unit-at-a-time granularity) and covered by a test, not just implemented and hoped.

## modeling.py + probe.py (2026-08-18)

- Approved interpretation applied: `allocation.py` stays pure; `modeling.py` derives `alpha_pattern`
  from `rank_pattern` + `scaling_mode` in `compute_alpha_pattern`, and `alloc_json_payload` merges it
  with the `Allocation` when writing metadata to disk.
- `rank_pattern`/`alpha_pattern` are fed to PEFT's `LoraConfig` keyed by the *pre-wrap* module names
  (as discovered from the plain base model before `get_peft_model()` runs) -- confirmed empirically
  that PEFT matches those during injection against the base tree it's wrapping, not the final tree.
- **Real bug caught only by actually running this against the live Qwen2.5-0.5B model, not by
  reasoning about the API**: `get_peft_model()` wraps every module under a `base_model.model.`
  prefix, so a first version of `verify_live_model` (and `probe.py`'s `lora_B` lookup) walked the
  *wrapped* model's `named_modules()` and compared those prefixed names directly against
  `rank_pattern` -- silent zero matches, not a crash, would have looked like "verification found no
  live LoRA layers" rather than "verification is comparing the wrong strings" if I hadn't printed and
  inspected an actual live name. Fixed by stripping the confirmed `base_model.model.` prefix
  (`modeling.PEFT_NAME_PREFIX`) before comparing, in both places. Documented as a docstring on
  `verify_live_model`, not just fixed silently, since anyone extending this code will hit the same
  trap.
- Installed `peft` (plus a real download of `Qwen/Qwen2.5-0.5B-Instruct`) into the local venv and did
  full CPU/fp32 smoke runs -- `discover_module_specs` finds all 168 real target Linear layers (24
  layers x 7 projections); `verify_live_model` passes for both a uniform allocation
  (`constant_ratio`, all ranks=16) and a skewed one (`fixed_alpha`, temperature=0.5, confirmed every
  live alpha == the fixed constant); `run_probe` completes end-to-end for both `gsm8k` and `alpaca`,
  writes valid JSON to `results/probe/<probe_id>.json`, and produces non-negative signal values for
  all four keys (`rms`, `raw_norm`, `fisher`, `relative`) across all 168 modules. `scripts/run_probe.py`
  CLI wired and smoke-tested end-to-end (`--device cpu`, tiny `probe.rank`/`probe.steps` overrides).
- These were CPU/fp32/tiny-step smoke tests (per the CPU-vs-Kaggle split from the phase plan), not the
  real fp16/CUDA/gradient-checkpointing/probe_steps=100 path -- that's the actual P3 gate, and it
  needs a real GPU. Added `OptimConfig.micro_batch` (default 4) using the real value P0's preflight
  calibrated on the T4, since both `run_probe.py` and the coming `train.py` need it.

## modeling.py + probe.py Kaggle gate confirmation (2026-08-18)

**Gate**: PASSED. Kaggle (Tesla T4, fp16/CUDA, after re-running `pip install -r requirements.txt` --
Kaggle sessions don't persist installed packages, this tripped the same torchao issue again on a
fresh session, not a code regression): live-model verification for the uniform allocation matched
exactly (`adapter_params_verified` total 8798208 == `alloc.params_total` 8798208). Both probes
(`gsm8k`, `alpaca`) ran the real `probe_rank=8`/`probe_steps=100` and wrote valid JSON in ~40s and
~36s respectively -- close to and under the ~108s (`2 x 100 steps`) P0 projected for both probes
combined. One benign warning worth recording, not fixing: fp16 `GradScaler` occasionally skips
`optimizer.step()` on an inf/nan-gradient step, but `probe.py` calls `scheduler.step()`
unconditionally, producing PyTorch's standard "`lr_scheduler.step()` before `optimizer.step()`"
warning. Harmless for the probe specifically -- the measured quantity is per-step gradient norms on
`lora_B`, not schedule fidelity -- so not worth the added complexity of conditioning the scheduler
step on scaler success for a diagnostic-only training loop. Would need addressing for real if it
showed up in the main `train.py` loop, where the LR schedule's shape actually matters.

## train.py + evaluate.py + run_single.py (2026-08-20)

- `train.py`: hand-written loop, fixed-step budget (I3), AdamW + cosine warmup, fp16 `GradScaler`.
  Properly guards the LR-scheduler-vs-optimizer-skip race this time (unlike `probe.py`, documented
  as intentionally not worth it there): tracks `scaler.get_scale()` before/after `scaler.step()` and
  only advances the scheduler when the optimizer actually stepped, since here the schedule shape is a
  real result, not just probe diagnostics. `step_log.jsonl` gets per-step cumulative GPU seconds via
  a `torch.cuda.Event` recorded every step (not just start/end) -- costs a sync per step, judged
  worth it for exact logging given the generous 2.63h tier-1 budget from P0. OOM caught internally
  and returned as `status="oom"` rather than raised, so a real run can fail one condition without
  taking down the whole grid (run_grid.py's job, not this module's, to keep going after that).
- `evaluate.py`: `loss_token_weighted`/`loss_example_mean` computed manually from logits (not
  `out.loss`) specifically so both per-token and per-example breakdowns are available from one
  forward pass. GSM8K `extract_strict`/`extract_flexible` + `Fraction`-based comparison, 15 new unit
  tests (`test_evaluate.py`, pure logic, no GPU) covering comma/currency/percent/decimal
  normalization and both extractors -- not in §7's list but cheap and worth having given how easy
  this kind of parsing is to get subtly wrong.
- `run_single.py` wires the whole pipeline (allocation -> model build -> live verification -> alloc.json
  -> training with periodic held-out-loss curve -> final held-out loss -> generation eval ->
  `results.csv` row + `metrics.json` + `samples.jsonl` + adapter-only save) and handles `strategy=
  "zero_shot"` as a distinct eval-only branch (no allocation, no LoRA, just the untuned base model),
  matching the tier-1 grid's eval-only condition.
- Full local smoke coverage on CPU (fp32, `--smoke`: 20 steps, 8 eval examples, `budget_rank=4`,
  `max_new_tokens=16` to keep it fast) for **both** branches: the trained path (`strategy=uniform`)
  and the `zero_shot` path. Both wrote complete run directories (`alloc.json`, `step_log.jsonl`,
  `held_out_loss_curve.json`, `samples.jsonl`, `metrics.json`, adapter-only checkpoint for the trained
  run) and correct `results.csv` rows, verified by inspection: `adapter_params_verified=True`,
  `budget_abs_error=0`, and directionally sane losses (zero_shot held-out loss 0.93 > trained
  held-out loss 0.59, i.e. 20 steps of training measurably helped, as expected). Full suite still
  254/254 green (239 prior + 15 new evaluate tests).
- These smoke runs are CPU/fp32/tiny-scale correctness checks only, not the timed gate. The actual
  §4.6 requirement -- `run_single.py --smoke` under 3 minutes -- needs the real fp16/CUDA path and
  can only be measured on Kaggle.

**Gate**: PASSED. Kaggle (Tesla T4, fp16/CUDA): `run_single.py --smoke` finished in 1m49.979s wall
(well under the 3-minute cap), `train_gpu_seconds=11.38` for 20 steps, `eval_gpu_seconds=28.34` for
held-out loss + 8-example generation eval, `peak_vram_mb=4557` (well within the T4's 16GB),
`adapter_params_verified=true`, `budget_abs_error=0`. `gsm8k_strict`/`gsm8k_flexible` both 0.375 (3/8)
-- directionally sane for 20 smoke-scale steps, not a claim about real performance. Full run
directory written correctly. Both code paths (trained + `zero_shot`) now confirmed working on real
fp16/CUDA, not just CPU smoke.

## metrics.py, run_grid.py, analyze.py, make_figures.py, configs/*.yaml (2026-08-20)

- `metrics.py` (§4.8, not assigned to an earlier phase but a prerequisite for `analyze.py`, so built
  here): `paired_delta`, `pooled_sd`, `minimum_detectable_effect`, `hedges_g`, `cliffs_delta`. No
  significance tests anywhere, by design. 11 new unit tests, all local/pure, including a known-value
  check on `pooled_sd` and monotonicity checks on the MDE formula.
- `configs/base.yaml`, `grid_tier1.yaml`, `grid_tier2.yaml` created (§3 lists them but no earlier
  phase built them). Tier 1 matches §6 exactly: 16 training runs + 1 eval-only (`zero_shot`) + 2
  probes. Tier 2 matches §6's six arms in order; the AdaLoRA reference arm is deliberately **not**
  included as a grid entry -- it needs non-parameter-matched handling `run_single.py`'s pipeline
  doesn't support, and the spec explicitly calls it the most droppable item.
- `run_grid.py`: each run executes as an isolated `run_single.py`/`run_probe.py` subprocess rather
  than in-process, specifically so 16+ sequential model loads don't accumulate CUDA memory
  fragmentation in one long-lived process, and so a crash in one run can't corrupt state for the
  next. `results.csv` itself is written by `run_single.py`'s own atomic append; `run_grid.py` only
  decides what to run/skip (via `run_id`, computed identically without needing to execute anything)
  and writes a synthetic `status=failed` row only if a subprocess dies before writing its own.
  `--dry_run` needs zero GPU work (no model ever loads -- `run_id`/`probe_id` are pure hashes of the
  resolved config), so it's fully verifiable locally.
- Grid-level `temperature: [0.5, 2.0]`-style list overrides (tier 2's temperature sweep) expand via
  `itertools.product` over any list-valued per-condition key, multiplied by `n_seeds` -- generic
  enough for the one sweep the spec actually uses, not built out further than that.
- Probe scheduling: any grid whose conditions include `gradnorm_prop`/`gradnorm_inverse` gets the
  GSM8K probe auto-added to the run list even if the grid's own `probes:` field omits it (tier 2's
  `probes: []`, since it wants to *reuse* tier 1's already-cached probe, not re-run it) --
  `probe_id` is a pure hash of `(model_name, task, rank, steps, seed=0, split_seed)`, so tier 2 and
  tier 1 independently compute the identical id from the same `base.yaml` and the cache hits. Only
  the GSM8K probe drives allocation; the Alpaca probe exists purely for Figure 1's diagnostic
  comparison and feeds no strategy.
- **Resume/SIGKILL verified for real, locally, not simulated**: built a tiny throwaway grid (3
  strategies x a few seeds, 5 steps, CPU) outside the repo, launched `run_grid.py` as a real
  background process, and sent it a genuine `kill -9` mid-grid (confirmed via `ps` that the target
  PID was alive at kill time, and that it was actively past the first couple of runs). Restarting
  with the identical grid produced **zero duplicate `run_id`s and zero lost rows** -- every run
  from both before and after the kill ended up in `results.csv` exactly once. This is a
  process/file-integrity property, not a GPU-numerics one, so local CPU verification is a genuine
  test of the real mechanism, not a stand-in for one.
- `analyze.py`: paired deltas + MDE-gated read of BUILD_SPEC.md §4.4's four-outcome interpretation
  table, computed per metric (`loss_token_weighted`, lower-is-better; `gsm8k_flexible`,
  higher-is-better) via a `LOWER_IS_BETTER` sign flag. Explicitly documented as a coarse read to
  pair with the raw `comparisons` dict, not a verdict to trust blindly -- BUILD_SPEC.md §4.8 is
  clear that this pipeline computes no significance tests anywhere, and n<=5 seeds means the
  MDE-gated label is a starting point for a human reading the real numbers, not the final word.
  Smoke-tested against local synthetic `results.csv` (no crash; correctly reports "insufficient
  conditions" when `gradnorm_prop`/`gradnorm_inverse` rows don't exist yet).
- `make_figures.py`: all six figures implemented, each skips gracefully (prints why, doesn't crash)
  when its required data isn't present yet rather than assuming real tier-1 data exists. Figure 1
  (allocation profiles) reads probe JSON directly, not `results.csv`, since per-module signal data
  has nowhere else to live -- "regenerable from results.csv alone" is read as "from the saved
  `results/` tree, without re-running experiments," not literally only the CSV file. Rendered and
  visually inspected 5 of 6 locally against synthetic/smoke data (missing only the `fixed_alpha`
  scaling-trap panel, since no such condition exists in my tiny local test set); all renders were
  structurally sane, not garbled or empty.
- Full suite: 265/265 green (254 prior + 11 new metrics tests).

**Gate**: PASSED. `--dry_run` on both real tier-1 and tier-2 grids verified locally (correct run
counts, correct probe-id reuse/cache-skip, correct skip-on-resume); SIGKILL/resume verified for real
locally (process/file-integrity property, not GPU-numerics). Kaggle `--dry_run` against
`grid_tier1.yaml` after a fresh `preflight.py` run: both probes correctly showed `SKIP (cached)`
(reusing the P3-gate probe JSONs already on disk -- confirms `probe_id` determinism holds across
sessions, not just within one process), all 17 condition-runs listed correctly, projected 2.60
GPU-hours for the pending work -- consistent with `preflight.py`'s own 2.64h tier-1 projection (both
use the same coarse eval-time multiplier methodology), well under the 8h cutoff. Every piece of P5
now confirmed against the real environment, not just local synthetic data.

## README (2026-08-20)

- Quickstart leads with the Kaggle "Internet on" caveat per §2, since that's the single most common
  first-run failure and it's easy to bury in a longer doc.
- Every command in the README is one already run for real somewhere in this log (preflight, smoke,
  probe, single-condition, grid `--dry_run`/real, analyze, make_figures, pytest) -- nothing here is
  speculative or untested.
- Definition-of-done section maps each §8 checklist item to the exact command that verifies it,
  including the two that are automatic/structural rather than something you eyeball: budget error
  via a one-line pandas read of `results.csv`, and parameter verification via the `assert` already
  inside `run_single.py` (it fails the run loudly, not silently, if it's ever wrong).
- Full suite re-run one final time before this commit: 265/265 green.

**Gate**: `pytest -q` 265/265 green (final check, no GPU needed for the suite itself). README covers
every command needed to reproduce every number and figure. The real tier-1/tier-2 grid runs
themselves haven't been executed yet -- that's the repo owner's call on timing, not part of this
build's phase gates, and `results/` is deliberately gitignored so no partial/fake experiment data
ships in the repo.

## Bug found during the real tier-1 run: fp16 GradScaler overflow poisons probe signals (2026-08-20)

- Repo owner ran the real tier-1 grid on Kaggle: `zero_shot`, `uniform` (5 seeds), `random` (3
  seeds) all succeeded; all 5 `gradnorm_prop` and all 3 `gradnorm_inverse` runs failed identically:
  `ValueError: cannot convert float NaN to integer` inside `solve_allocation`'s
  `math.floor(ideal[n])`.
- Root cause is in `probe.py`, not `allocation.py`. fp16 `GradScaler` routinely detects an
  overflowed gradient on an early step (this is normal -- the whole point of dynamic loss scaling)
  and skips `optimizer.step()` for that step. `probe.py` was reading `lora_B.weight.grad` and
  folding it into the running signal average *before* checking whether the scaler actually accepted
  the step -- so one overflow step (inf/nan gradient) silently poisoned that module's averaged `rms`
  signal with NaN for the entire probe run. `train.py` already guarded against exactly this failure
  mode (`scale_before`/`scale_after` comparison gating `scheduler.step()`); `probe.py` never did,
  and CPU smoke testing couldn't have caught it since there's no GradScaler/overflow concept without
  real fp16 autocast -- this needed a real fp16 GPU run to surface, which it did, on the first real
  grid run.
- Fixed by computing each step's per-module contribution into a temporary dict, then only merging it
  into the running `accum` totals (and only calling `scheduler.step()`, and only incrementing a new
  `valid_steps` counter) once the scaler confirms the step was actually applied. The final signal
  average now divides by `valid_steps`, not the requested `steps` -- if a couple of early steps get
  skipped that's fine and expected, the divisor stays correct. `valid_steps` is now recorded in the
  probe JSON output for transparency. Added an `assert valid_steps > 0` for the (very unlikely)
  degenerate case where every step overflows, since that would point at a real lr/init problem, not
  routine overflow.
- Defense in depth: `solve_allocation` now explicitly rejects non-finite or negative weights with a
  clear error message pointing at a corrupted probe signal, instead of failing three calls deep
  inside `math.floor` with a generic `ValueError` that gives no hint where to look.
- **Operational consequence for the repo owner, not just a code fix**: the probe JSON already cached
  on Kaggle disk (`results/probe/0ea0f59b6079.json`) was generated *before* this fix and still
  contains the NaN-poisoned signal. `probe_id` hashes the probe *config*, not its contents, so
  `run_grid.py`'s cache check (`if probe_path.exists(): skip`) will keep reusing that stale, broken
  file indefinitely unless it's deleted first. Told the repo owner to delete it before retrying.
- Separately (found by code inspection while diagnosing this, not from a second Kaggle failure): once
  `run_grid.py` writes a `status=failed` row for a crashed run, that `run_id` was permanently
  un-retryable on every future invocation, since `existing_run_ids` treated any row -- regardless of
  status -- as "already done, skip". A fix that only changes the code does nothing for a condition
  that's already marked failed in `results.csv`, since the grid would just keep skipping it forever.
  Fixed `io_utils.existing_run_ids` to accept `exclude_statuses`, and `run_grid.py` now passes
  `exclude_statuses={"failed"}` so a failed run is retried on the next invocation, while `ok`/`oom`
  rows still count as done (retrying an OOM at the same batch size would just OOM again, per §4.6's
  "do not silently retry at a different batch size").
- Full suite: 265/265 still green. Regression-checked the CPU probe path specifically (unaffected,
  since there's no scaler-skip concept without real fp16 -- `valid_steps == steps` there always).

**Gate**: PASSED. Repo owner deleted the stale probe (twice -- first deleted the wrong files,
`results/runs/<run_id>/` instead of `results/probe/<probe_id>.json`, easy mix-up since both are
12-hex-char names; the second attempt hit the exact stale-cache failure predicted, confirmed by the
new clear error message from the `solve_allocation` defensive check rather than a cryptic
`math.floor` crash -- a real, useful signal that the guard is working) and re-ran the real tier-1
grid on Kaggle. All 17 condition-runs succeeded this time, including all 8 `gradnorm_prop`/
`gradnorm_inverse` runs that failed before the fix. The retry-after-failure fix also confirmed
correct in the wild: the 9 already-`ok` rows were skipped, the 8 `failed` rows were retried, exactly
as intended.

## Real tier-1 result and a second analyze.py bug (2026-08-20)

- With the full tier-1 grid now complete, the repo owner ran `scripts/analyze.py` for real and asked
  me to walk through what it meant. Reading the actual numbers: on `loss_token_weighted` (primary
  metric), all three non-uniform strategies are **reliably worse than uniform**, each clearing the
  MDE (0.000663) in the bad direction -- `random` +0.00082 (1.2x MDE), `gradnorm_inverse` +0.00131
  (2.0x MDE), `gradnorm_prop` +0.00268 (**4.0x MDE**, the worst of the three). Cliff's delta is 1.0
  for both gradnorm arms vs uniform: complete separation, every seed pair worse, not just the mean.
  `gradnorm_prop` -- the hypothesis arm -- is the worst performer of all, worse than its own
  directional control (`gradnorm_inverse`). On `gsm8k_flexible` (secondary), none of the deltas clear
  that metric's much larger MDE (0.0389), so "doesn't matter at this scale" is the correct read there.
- **`analyze.py`'s `interpretation` printed "allocation doesn't matter at this scale" for the loss
  metric too, which was wrong** -- a real bug I found by actually reading the numbers against the
  label, not by re-deriving the whole pipeline. `_beats_uniform` only tested "does this strategy beat
  uniform" (a boolean), so "reliably worse than uniform beyond the MDE" and "no detectable effect
  either way" collapsed into the same negative case and got the same label. Those are different
  findings -- a clean negative result (everything tested is worse than the baseline, consistently) is
  not the same as a null result (nothing detectable in either direction), and BUILD_SPEC.md's
  four-outcome table doesn't name the "uniform wins outright" case at all.
- Fixed by replacing `_beats_uniform` (bool) with `_direction` (`"better"`/`"worse"`/`"noise"`,
  MDE-gated in both directions), and adding a fifth interpretation branch ahead of the four spec'd
  ones: all three controls `"worse"` -> explicit "uniform beats every non-uniform strategy tested...
  this is a real negative result, not 'allocation doesn't matter'". Verified directly against the
  repo owner's actual pasted numbers before committing (reconstructed the `comparisons` dict, called
  `interpretation()` locally, confirmed it now prints the corrected read for both metrics). Full
  suite: 265/265 still green.
- Take the tier-1 headline as it now stands: **uniform allocation beat every alternative tested on
  held-out loss**, and the more directed the reallocation (prop > inverse > random, in order of how
  much worse each got), the worse it performed -- the opposite of the hypothesis. Worth investigating
  *why* before writing this up -- candidates worth checking: whether 400 steps is enough for
  reallocated capacity to pay off, whether `budget_rank=16` leaves too little room for reallocation to
  matter, or something about the `rms` probe signal itself -- but that investigation is the repo
  owner's call, not something to guess at here.

**Gate**: N/A (bugfix, not a phase). `pytest -q` 265/265 green; fix verified against real pasted
results before pushing, not just synthetic data.

## Second analyze.py bug: blanket MDE ignored unequal seed counts (2026-08-20)

- Repo owner asked directly whether the unequal seed counts tier 1 uses by design (`uniform`/
  `gradnorm_prop` at 5 seeds, `gradnorm_inverse`/`random` at 3, per §6) affect the read -- a sharp
  question that exposed a second real bug: `analyze_metric` computed one `minimum_detectable_effect`
  using `n_uniform` (5) and applied it to every comparison, including the 3-seed ones. A 3-seed
  comparison has a coarser true detection threshold than a 5-seed one; using the 5-seed MDE for it
  understated how much noise there could be.
- Recomputed with the correct per-comparison n: `random`'s "worse than uniform" call (delta 0.00082
  vs the wrong blanket MDE 0.000663) does not survive against its own correct 3-seed MDE (0.000996)
  -- delta < MDE, so it's noise, not a real effect. `gradnorm_prop` (n=5, already used the right MDE)
  and `gradnorm_inverse` (delta 0.00131 vs correct MDE 0.000996) both still hold up as reliably worse
  than uniform.
- Fixed by moving MDE computation into `compare_to_uniform` itself, using each comparison's own
  `paired_delta` result's `n` (already correctly reflects only the matched seed pairs) against the
  shared noise-floor sigma. `interpretation()` now reads each comparison's own MDE instead of a single
  blanket value passed in. Verified against the real numbers again before pushing (random correctly
  flips from "worse" to "noise"; the interpretation string correctly falls into the "mixed directions"
  branch instead of overclaiming "uniform beats everything"). Full suite: 265/265 still green.
- Revised headline: `gradnorm_prop` and `gradnorm_inverse` are both reliably worse than uniform;
  `random` is not distinguishable from noise at n=3. Still not the hypothesis's direction, but weaker
  and more honest than "uniform beats every alternative tested" -- worth remembering that `random`'s
  own MDE is coarse specifically because of the 3-seed control-arm design, not because there's
  necessarily nothing there.

**Gate**: N/A (bugfix, not a phase). `pytest -q` 265/265 green.
