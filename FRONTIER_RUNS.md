# Frontier Runs

The repo is now split into three explicit lanes:

- `stable`: submission-facing, reviewer-friendly, promotion-eligible.
- `challenger`: opt-in only, manual review required, never submission-safe by default.
- `research_only`: separate runner under `research_only/`, never launched from `research/run.py`.

## Current Branch Of Record

The default branch-of-record is the SP8192 mainline family:

- `sp8192_mainline_base`
- `sp8192_mainline_recur345_par7`
- `sp8192_mainline_recur345_par7_qk525`
- `sp8192_mainline_recur345_par7_qk525_ttt`
- `sp8192_mainline_submit_safe`

Use these for stable-lane work. They are the only presets intended for automatic promotion and final submission packaging.

## RunPod Quickstart

For a clean H100 pod flow, use the wrapper script instead of manually chaining install, data, and launcher commands:

```bash
bash scripts/runpod_frontier.sh prep
bash scripts/runpod_frontier.sh smoke
bash scripts/runpod_frontier.sh train
bash scripts/runpod_frontier.sh all
```

Notes:

- The wrapper hard-resets the repo checkout to `origin/<branch>` before each run stage.
- It also wipes stale run outputs, the selected variant dataset/tokenizer artifacts, `data/manifest.json`, `docs_selected.*`, and temporary pip caches inside repo/tmp boundaries before rebuilding the pod state.
- On a fresh pod it reruns the approved cloud installer, verifies that both `sentencepiece` and `flash_attn` import, installs them with the approved pip command if they do not, and only then runs the hard frontier environment gate.
- It fails fast on missing CUDA/FlashAttention readiness, missing dataset shards, or missing tokenizer artifacts.
- If you do not pass `--variant`, the script resolves a preset-compatible variant for `smoke`, `train`, and `all` so the stable SP8192 defaults stay launchable.
- Ephemeral pods should use published cached artifacts by default. If the cached manifest does not publish the requested variant, prep now stops quickly by default with the missing dataset/tokenizer status and the exact manual bootstrap command.
- Docs bootstrap is heavyweight and opt-in only via `--allow-docs-bootstrap`. It runs only after a conservative disk-space preflight and warns if `HF_TOKEN` is unset.
- For large Hugging Face transfers, set `HF_TOKEN` before retrying any docs bootstrap path.

## Stable Lane

Stable-lane rules:

- no prefix matcher
- no pre-quant TTT
- no challenger-only hooks
- canonical reporting only
- `submission_safe=true` is possible only here

Recommended commands:

```bash
python3 research/run.py --preset sp8192_mainline_base --scale smoke --run-name sp8192_mainline_base_smoke --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset sp8192_mainline_recur345_par7_qk525_ttt --scale half_run --run-name sp8192_mainline_half --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset sp8192_mainline_submit_safe --scale submit_safe --run-name sp8192_mainline_submit --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

Equivalent wrapper for the stable mainline submit-safe launch:

```bash
bash scripts/runpod_frontier.sh train --variant sp8192 --train-preset sp8192_mainline_submit_safe
```

Quantization and tokenizer surfaces:

- `quant_grouped_sdclip`
- `quant_calib_mixed`
- `quant_embed_split_policy`
- `tok_sp8192_clean`
- `tok_sp7680_clean`
- `tok_sp7168_clean`

## Challenger Lane

Challenger-lane rules:

- must be launched with `--allow-challenger`
- always emits `manual_review_required=true`
- never emits `submission_safe=true`
- always writes a `rule_audit.{json,txt}` artifact

Available challenger presets:

- `challenger_prefix_matcher`
- `challenger_prefix_matcher_ttt`
- `challenger_prequant_ttt`
- `challenger_prefix_matcher_prequant_ttt`

Example:

```bash
python3 research/run.py --preset challenger_prefix_matcher --scale half_run --run-name challenger_prefix_half --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100 --allow-challenger
```

## Run Artifacts

Every completed run now records:

- training-best validation BPB
- official post-quant submission BPB
- artifact bytes and remaining headroom
- train time and official eval time
- quant gap when available
- stable/challenger lane metadata
- deltas versus the `2026-04-11` merged and live leaderboard bars

Use these to decide whether to promote, discard, or escalate a branch.

## Submission Readiness

Check the latest stable candidate:

```bash
python3 scripts/submission_readiness.py --latest --family frontier --lane stable --require-submission-safe
```

Backfill an older run:

```bash
python3 scripts/submission_readiness.py --run-dir research/results/runs/<timestamp_run_name> --rewrite
```

## Legacy Appendix

The older cache / Dirichlet / frontier-control stack is still present but explicitly demoted to `legacy`:

- `control_verified_sota`
- `sota_plus_ngram7`
- `sota_plus_ppm_multiorder`
- `sota_plus_ppm_entropy_fixed`
- `sota_plus_ppm_entropy_order_adaptive`
- `sota_plus_ppm_dirichlet`
- `sota_plus_ppm_dirichlet_submit`
- `xsaall_fullgptq_prune_plus_cache`
- `rotaryfix_bigram3072_legalttt`

These remain useful as references and rollback points, but they are not the default branch-of-record for new leaderboard work.
