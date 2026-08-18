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
