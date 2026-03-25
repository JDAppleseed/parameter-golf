# Frontier Runs

This repo is now wired for five legal frontier presets through [`research/run.py`](/Users/jonathandavanzo/Desktop/parameter-golf/research/run.py).

Canonical control:

- `control_verified_sota`
- Trainer: [`train_gpt_frontier_control.py`](/Users/jonathandavanzo/Desktop/parameter-golf/train_gpt_frontier_control.py)
- This is the March 23 verified control family: LeakyReLU^2 + legal score-first TTT + Parallel Muon.
- Treat this as the benchmark reference for all serious 8xH100 comparisons.

Important legality and reproducibility notes:

- Cache presets use a strict score-then-commit interface from [`frontier_cache.py`](/Users/jonathandavanzo/Desktop/parameter-golf/frontier_cache.py). A scored segment cannot be committed early, and the next segment cannot be scored until the prior one is committed.
- TTT remains score-first: each chunk is scored before any chunk-local adaptation step.
- Cache-enabled distributed eval uses a canonical rank-0 evaluation path, then broadcasts metrics. This is slower than fully sharded eval, but it keeps cache semantics deterministic and legal.
- `artifact_bytes` are measured from the counted code bundle plus the exported compressed model blob. The checkpoint helper in [`frontier_checkpoint.py`](/Users/jonathandavanzo/Desktop/parameter-golf/frontier_checkpoint.py) is included in `code_bytes`.
- Exact frontier presets remain CUDA-only. For Apple Silicon smoke work, use the existing MLX proxy presets, not these final frontier trainers.

## Configs

| Preset | What it tests | Risk |
| --- | --- | --- |
| `control_verified_sota` | Verified control branch, no eval cache | green |
| `sota_plus_ngram7` | Verified control + deterministic 7-gram backward-looking cache | green |
| `sota_plus_ppm_multiorder` | Verified control + deterministic multi-order PPM-style backoff cache | yellow |
| `xsaall_fullgptq_prune_plus_cache` | XSA-all + full GPTQ + selective pruning + causal cache | yellow |
| `rotaryfix_bigram3072_legalttt` | RotaryFix + BIGRAM3072 + legal score-first TTT | green |

## Commands

`control_verified_sota`

```bash
python3 research/run.py --preset control_verified_sota --scale smoke --run-name control_verified_sota_smoke_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset control_verified_sota --scale half_run --run-name control_verified_sota_half_run_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset control_verified_sota --scale full_run --run-name control_verified_sota_full_run_s1337 --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

`sota_plus_ngram7`

```bash
python3 research/run.py --preset sota_plus_ngram7 --scale smoke --run-name sota_plus_ngram7_smoke_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset sota_plus_ngram7 --scale half_run --run-name sota_plus_ngram7_half_run_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset sota_plus_ngram7 --scale full_run --run-name sota_plus_ngram7_full_run_s1337 --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

`sota_plus_ppm_multiorder`

```bash
python3 research/run.py --preset sota_plus_ppm_multiorder --scale smoke --run-name sota_plus_ppm_multiorder_smoke_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset sota_plus_ppm_multiorder --scale half_run --run-name sota_plus_ppm_multiorder_half_run_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset sota_plus_ppm_multiorder --scale full_run --run-name sota_plus_ppm_multiorder_full_run_s1337 --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

`xsaall_fullgptq_prune_plus_cache`

```bash
python3 research/run.py --preset xsaall_fullgptq_prune_plus_cache --scale smoke --run-name xsaall_fullgptq_prune_plus_cache_smoke_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset xsaall_fullgptq_prune_plus_cache --scale half_run --run-name xsaall_fullgptq_prune_plus_cache_half_run_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset xsaall_fullgptq_prune_plus_cache --scale full_run --run-name xsaall_fullgptq_prune_plus_cache_full_run_s1337 --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

`rotaryfix_bigram3072_legalttt`

```bash
python3 research/run.py --preset rotaryfix_bigram3072_legalttt --scale smoke --run-name rotaryfix_bigram3072_legalttt_smoke_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset rotaryfix_bigram3072_legalttt --scale half_run --run-name rotaryfix_bigram3072_legalttt_half_run_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset rotaryfix_bigram3072_legalttt --scale full_run --run-name rotaryfix_bigram3072_legalttt_full_run_s1337 --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

Resume an interrupted run:

```bash
python3 research/run.py --resume-run-dir research/results/runs/<timestamp_run_name>
```

Compare runs:

```bash
python3 research/compare_runs.py --family frontier --status all --limit 20
python3 research/compare_runs.py --family frontier --scale half_run --status all --json
```

Inspect progress:

```bash
LATEST_RUN="$(ls -td research/results/runs/* | head -n 1)"
cat "$LATEST_RUN/run_summary.json"
tail -f "$LATEST_RUN/launcher.log"
```

Inspect byte budget for a completed run:

```bash
LATEST_RUN="$(ls -td research/results/runs/* | head -n 1)"
cat "$LATEST_RUN/byte_budget.txt"
cat "$LATEST_RUN/legality_note.txt"
```

## Promotion Ladder

Recommended budget-first workflow:

1. Run `control_verified_sota` at `half_run` on 1xH100.
2. Run `sota_plus_ngram7` at `half_run` on the same box and seed.
3. Compare those two first.
4. Promote only the better of those two immediately.
5. If the cache branch is slow, unstable, or legality-review-hostile, run `rotaryfix_bigram3072_legalttt` next.
6. Run `sota_plus_ppm_multiorder` only if `sota_plus_ngram7` is promising and cache overhead looks acceptable.
7. Keep `xsaall_fullgptq_prune_plus_cache` as a high-upside yellow branch after one green finalist is already identified.

Recommended full-run shortlist:

- Best of `control_verified_sota` vs `sota_plus_ngram7`
- `rotaryfix_bigram3072_legalttt` if cache overhead is questionable
- `xsaall_fullgptq_prune_plus_cache` only if cheap screening is clean and byte headroom remains healthy

## Kill Criteria

Stop or reject a branch immediately if any of these fire:

- `legality_note.json` status is not `legal`
- `byte_budget.json` reports `artifact_bytes_measured > 16000000`
- Cache branch `eval_time_seconds` is worse than the control by more than about 1.5x at the same scale
- Half-run `val_bpb` is worse than the half-run control by more than `0.005`
- Export fails, checkpoint resume fails, or the final eval label falls back to an incomplete/non-final metric

Soft yellow-branch kills:

- `xsaall_fullgptq_prune_plus_cache` has weak half-run signal and poor byte headroom at the same time
- `sota_plus_ppm_multiorder` is not clearly better than `sota_plus_ngram7`

## Outputs

Every completed run directory now includes:

- `run_spec.json`
- `result.json`
- `run_summary.json`
- `legality_note.json`
- `legality_note.txt`
- `byte_budget.json`
- `byte_budget.txt`
- `checkpoint_latest.pt` when checkpointing is enabled

The byte budget report is emitted automatically after a successful export and includes:

- `param_count`
- `raw_bytes_fp16_est`
- `raw_bytes_fp32_est`
- `post_quant_bytes_est`
- `exported_bytes_measured`
- `code_bytes_measured`
- `artifact_bytes_measured`
- `remaining_headroom_to_16MB`

## Recommendation

If you want one branch to get the first expensive 8xH100 full run after cheap screening, make it `sota_plus_ngram7`.

Reason:

- it is the narrowest high-upside extension on top of the verified control
- it keeps the legal story simple
- it is materially less brittle than the full GPTQ branch
- it gives you the cleanest read on whether backward-looking cache ideas deserve more spend
