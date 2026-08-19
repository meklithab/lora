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
