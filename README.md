# rankalloc

At a fixed total LoRA adapter parameter count, does non-uniform layer-wise rank allocation beat
uniform allocation? `Qwen/Qwen2.5-0.5B-Instruct` on GSM8K, built for a free Kaggle/Colab T4.

See [`BUILD_SPEC.md`](BUILD_SPEC.md) for the full experiment design and the three invariants (I1
parameter matching, I2 scaling invariance, I3 compute matching) every result here is checked against,
[`CLAUDE.md`](CLAUDE.md) for the working rules, and [`NOTES.md`](NOTES.md) for the running log of
decisions, deviations, and every phase gate's result.

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
python scripts/preflight.py                 # CUDA check, downloads, micro-batch calibration,
                                              # tier-1 GPU-hour projection -> results/preflight.json
python scripts/run_single.py --smoke         # full pipeline on 20 steps / 8 eval examples, <3 min
```

If `preflight.py` projects tier 1 over 8 GPU-hours, cut `optim.max_steps` or the train subset in
`configs/base.yaml` before running the real grid -- that's a deliberate human decision point, not
something any script decides on its own.

## Running the experiment

**A single condition**, e.g. a `gradnorm_prop` run against an already-computed probe:

```bash
python scripts/run_probe.py --set probe.task=gsm8k --set probe.rank=8 --set probe.steps=100
python scripts/run_single.py --condition demo \
  --set alloc.strategy=gradnorm_prop --set alloc.probe_id=<probe_id printed above>
```

Every `--set key.path=value` dotted override applies on top of `configs/base.yaml` (or whatever
`--config` points at); see [`src/rankalloc/config.py`](src/rankalloc/config.py) for the full schema.
An unknown key in a YAML config or a `--set` path is a hard error by design -- a silently-defaulted
typo is how a grid produces garbage.

**The full tier-1 grid** (16 training runs + 1 eval-only + 2 probes -- the runs that make the paper):

```bash
python scripts/run_grid.py --grid configs/grid_tier1.yaml --dry_run   # lists every run + projected
                                                                        # GPU-hours, no GPU work done
python scripts/run_grid.py --grid configs/grid_tier1.yaml             # the real run
```

Re-running the same command **resumes automatically, no flags needed**: `run_grid.py` skips any
`run_id` already present in `results/results.csv`. Kaggle/Colab sessions die without warning, so this
is the only thing that makes the grid survivable -- if a session dies mid-grid, just re-run the same
command in a fresh session after `pip install -r requirements.txt` again (packages don't persist
between Kaggle sessions, but `results/` does if it's inside your working directory / persisted
output). Add `--max_hours H` to stop launching new runs once your invocation has spent `H` GPU-hours,
useful for splitting a long grid across multiple sessions deliberately.

**Tier 2** (only start if >=12h remain after tier 1 -- see `BUILD_SPEC.md` §6 for why each arm exists
and the order to run them in):

```bash
python scripts/run_grid.py --grid configs/grid_tier2.yaml --dry_run
python scripts/run_grid.py --grid configs/grid_tier2.yaml
```

Tier 2's `gradnorm_prop` arms reuse tier 1's GSM8K probe automatically (same `probe_id`, computed
identically from `configs/base.yaml` regardless of which grid file is running) -- you don't need to
re-run the probe.

## Analysis and figures

Both read only the `results/` tree; neither re-runs anything:

```bash
python scripts/analyze.py                    # paired deltas, noise floor, MDE, effect sizes,
                                               # the four-outcome interpretation table (§4.4)
                                               # -> results/analysis.json (and stdout)
python scripts/make_figures.py                # all six figures -> results/figures/*.png
```

`analyze.py`'s "interpretation" string is a coarse MDE-gated read meant to sit alongside the raw
per-strategy deltas it also prints -- this pipeline deliberately computes **no significance tests**
anywhere (BUILD_SPEC.md §4.8: at n<=5 seeds the Wilcoxon floor exceeds 0.05 regardless of effect
size), so treat it as a starting point for reading the real numbers, not a verdict.

`make_figures.py`'s six figures:

1. Allocation profiles (probe rms signal by layer, GSM8K vs Alpaca side by side)
2. Forest plot (paired delta vs uniform per condition, individual seeds, MDE band)
3. Learning curves (held-out loss vs cumulative GPU-seconds, by condition)
4. Noise floor (the uniform-baseline seed spread)
5. Scaling-trap panel (`constant_ratio` vs `fixed_alpha` at identical allocations -- tier 2)
6. Budget fidelity (realised vs target parameter count per condition -- the proof of I1)

Each function skips gracefully (prints why) if its required data isn't in `results/` yet, rather than
assuming a full grid has already run.

## Tests

```bash
pip install -r requirements.txt   # pytest, and everything test_masking.py needs a real tokenizer for
pytest -q
```

No GPU needed -- `test_masking.py` downloads the real `Qwen/Qwen2.5-0.5B-Instruct` tokenizer and real
GSM8K/Alpaca data but only tokenizes (no model forward pass). 265 tests as of P5, covering
`allocation.py`'s invariants (§7, heaviest), response-only masking, config strict-validation and
`run_id` hashing, the GSM8K answer extractors, and `metrics.py`.

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

## Definition of done (BUILD_SPEC.md §8)

- `pytest` green -- `pytest -q`
- Smoke run under 3 minutes -- `time python scripts/run_single.py --smoke` (needs a real GPU to
  actually meet the 3-minute bar; verified 1m50s on a Tesla T4)
- Resume survives `SIGKILL` -- kill `run_grid.py` mid-grid, re-run the same command, check
  `results/results.csv` for duplicate or missing `run_id`s (there should be none)
- Every condition's `budget_rel_error` < 0.5% -- `python -c "import pandas as pd; df =
  pd.read_csv('results/results.csv'); print(df[['run_id','strategy','budget_rel_error']])"`
- `adapter_params_verified` matches `alloc.json` for all runs -- checked automatically inside
  `run_single.py` itself (an `assert` before training starts, not a post-hoc check)
- All six figures regenerate from the `results/` tree alone -- `python scripts/make_figures.py`
