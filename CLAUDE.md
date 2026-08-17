Working rules for this repository. `BUILD_SPEC.md` is the full specification; this file is the part that must survive context compaction. Re-read it before any non-trivial change.
What this repo is
A controlled experiment: at a fixed total LoRA adapter parameter count, does non-uniform layer-wise rank allocation beat uniform allocation? Research code under a hard deadline. Correctness of the comparison beats features, speed, and elegance.
The three invariants — never break these
I1 — Parameter matching. Adapter cost is `r · (d_in + d_out)`. Match `Σ r_m · c_m` across conditions, not `Σ r_m`. This model uses GQA (`k_proj` 896→128, `gate_proj` 896→4864), so equal rank sums give wildly unequal parameter counts.
I2 — Scaling invariance. LoRA scales by `α/r`. Default `scaling_mode="constant_ratio"` sets `α_m = 2·r_m` so `α_m/r_m` is constant across layers. Varying rank with fixed global `α` would change per-layer effective step size and confound capacity with learning rate. Scaling mode is identical across all conditions inside any one comparison.
I3 — Compute matching. Fixed optimizer-step budget, same data in the same order for every condition. Never epochs. Never "until converged".
Every one of these has an assertion in code and a test. If you're about to weaken one to make something work, stop and tell me instead.
Hard rules

* Never edit, loosen, delete, or `xfail` a test to make it pass. Fix the implementation, or tell me the spec is wrong.
* Never adapt batch size at runtime. Micro-batch is fixed by `preflight.py` for the whole grid. A run that OOMs fails loudly with `status=oom`.
* `raw_norm` is not the default probe signal. `rms` is. Raw gradient norm scales with module width, so it exists only as a deliberately-flawed comparison arm.
* Verify parameter counts against the live model, not the config dict. Walk `named_modules()`, read back `r` and `lora_alpha`, recompute from tensor shapes, assert against `alloc.json`.
* Fixed LR across all conditions. Per-condition tuning would destroy the comparison.
* No significance tests. At n≤5 the Wilcoxon floor exceeds 0.05 regardless of effect size. Report paired deltas, individual seed points, and the minimum detectable effect.
* `gradnorm_inverse` and `random` are mandatory arms, not nice-to-haves. Without them a positive result is uninterpretable.

Workflow

* Phase gates per `BUILD_SPEC.md` §8. Test, commit, stop and report at each gate.
* Append to `NOTES.md` every phase: decisions, surprises, deviations, timings.
* Flag anything that looks like >2 hours of work instead of building it.
* If a library API differs from what the spec assumes, stop and say so — don't invent an equivalent.

Environment
T4/P100, fp16 only (no bf16 on T4), `attn_implementation="sdpa"`, no FlashAttention-2, no bitsandbytes, no W&B. Kaggle needs "Internet on" for HuggingFace downloads. Sessions die without warning, so everything long-running resumes from `results.csv` with no flags.
Git
This repository currently holds an unrelated project. At P0, after I confirm I have a backup, wipe both its files and its entire commit history, then rename the repo to `rankalloc` (see `BUILD_SPEC.md` §9 for the exact commands). Do not run the wipe until I confirm.
Commits are authored as the repo owner via `git config user.name` / `user.email`. No `Co-Authored-By:` trailers, no generated-with attribution lines, no mention of AI assistance in any commit message or file. Use `scripts/dev/commit.sh "<ISO date>" "<message>"` for every commit — dates run 2026-08-17 to 2026-08-20 per the table in `BUILD_SPEC.md` §9. Branch per phase, merge `--no-ff`, `main` stays green.
