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

**Gate**: pending -- awaiting the repo owner's Kaggle run of `scripts/preflight.py` and the resulting
tier-1 GPU-hour projection before P1 starts.

### P0 follow-up: torchao incompatibility on Kaggle (2026-08-17)

First Kaggle run of `preflight.py` failed inside `get_peft_model()`: PEFT's LoRA layer dispatcher
eagerly calls `is_torchao_available()` for every module (unrelated to whether we use torchao), and
that function raises `ImportError` instead of returning `False` when it finds an incompatible
version -- Kaggle's preinstalled `torchao==0.10.0` trips it, even though nothing in this repo touches
torchao. Not a spec/invariant issue, a transitive dependency bug. Fixed by pinning `torchao>=0.16.0`
in `requirements.txt` so `pip install -r requirements.txt` upgrades it; BUILD_SPEC.md §2's pin list
doesn't mention torchao since we don't use it directly, so this is an addition, not a substitution.
Model download, both dataset downloads, and GPU/CUDA detection all succeeded before this point.
