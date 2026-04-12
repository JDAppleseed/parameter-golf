#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/runpod_frontier_common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/runpod_frontier.sh <prep|smoke|train|all> [options]

Options:
  --repo-dir PATH                 Repo checkout directory. Default: /workspace/parameter-golf
  --repo-url URL                  Repo URL. Default: https://github.com/JDAppleseed/parameter-golf.git
  --branch NAME                   Branch to hard-reset to. Default: main
  --variant NAME                  Dataset/tokenizer variant. Default: sp1024
  --train-shards N                Number of train shards to fetch. Default: 1
  --smoke-preset NAME             Smoke preset. Default: sp8192_mainline_base
  --train-preset NAME             Train preset. Default: sp8192_mainline_submit_safe
  --smoke-run-name NAME           Smoke run name. Default: sp1024_mainline_smoke
  --train-run-name NAME           Train run name. Default: sp8192_mainline_submit
  --seed N                        Seed. Default: 1337
  --smoke-gpu-profile NAME        Smoke gpu profile. Default: local_cuda
  --train-gpu-profile NAME        Train gpu profile. Default: 8xh100
  --skip-smoke                    Skip the 1-GPU smoke run before train/all
  --allow-docs-bootstrap          Allow heavyweight docs_selected.jsonl bootstrap when cached artifacts are unpublished
  --allow-missing-flash-attn      Relax only the final frontier env check for dry local/CPU edge cases
  -h, --help                      Show this help text
EOF
}

REPO_DIR="/workspace/parameter-golf"
REPO_URL="https://github.com/JDAppleseed/parameter-golf.git"
BRANCH="main"
VARIANT="sp1024"
VARIANT_EXPLICIT=0
TRAIN_SHARDS="1"
SMOKE_PRESET="sp8192_mainline_base"
TRAIN_PRESET="sp8192_mainline_submit_safe"
SMOKE_RUN_NAME="sp1024_mainline_smoke"
SMOKE_RUN_NAME_EXPLICIT=0
TRAIN_RUN_NAME="sp8192_mainline_submit"
TRAIN_RUN_NAME_EXPLICIT=0
SEED="1337"
SMOKE_GPU_PROFILE="local_cuda"
TRAIN_GPU_PROFILE="8xh100"
SKIP_SMOKE=0
ALLOW_DOCS_BOOTSTRAP=0
ALLOW_MISSING_FLASH_ATTN=0
STAGE=""
RESOLVED_VARIANT=""
RESOLVED_DATA_PATH=""
RESOLVED_TOKENIZER_PATH=""
RESOLVED_VOCAB_SIZE=""
PREP_DONE=0
PREP_PRESET=""
PREP_VARIANT=""
VARIANT_SOURCE=""
CACHED_DATASET_NAME=""
CACHED_DATASET_PRESENT=0
CACHED_TOKENIZER_NAME=""
CACHED_TOKENIZER_PRESENT=0
CACHED_MANIFEST_PATH=""
CACHED_REPO_ID=""
CACHED_REMOTE_ROOT_PREFIX=""
DOCS_BOOTSTRAP_MIN_FREE_BYTES=$((80 * 1024 * 1024 * 1024))
DOCS_BOOTSTRAP_WARN_FREE_BYTES=$((120 * 1024 * 1024 * 1024))

variant_vocab_size() {
  case "$1" in
    sp1024) printf '1024\n' ;;
    sp8192) printf '8192\n' ;;
    sp7680) printf '7680\n' ;;
    sp7168) printf '7168\n' ;;
    *) die "Unsupported variant '$1'. Expected one of: sp1024, sp8192, sp7680, sp7168." ;;
  esac
}

resolve_variant_paths() {
  local variant="$1"
  case "$variant" in
    sp1024)
      RESOLVED_DATA_PATH="$REPO_DIR/data/datasets/fineweb10B_sp1024"
      RESOLVED_TOKENIZER_PATH="$REPO_DIR/data/tokenizers/fineweb_1024_bpe.model"
      ;;
    sp8192)
      RESOLVED_DATA_PATH="$REPO_DIR/data/datasets/fineweb10B_sp8192"
      RESOLVED_TOKENIZER_PATH="$REPO_DIR/data/tokenizers/fineweb_8192_bpe.model"
      ;;
    sp7680)
      RESOLVED_DATA_PATH="$REPO_DIR/data/datasets/fineweb10B_sp7680"
      RESOLVED_TOKENIZER_PATH="$REPO_DIR/data/tokenizers/fineweb_7680_bpe.model"
      ;;
    sp7168)
      RESOLVED_DATA_PATH="$REPO_DIR/data/datasets/fineweb10B_sp7168"
      RESOLVED_TOKENIZER_PATH="$REPO_DIR/data/tokenizers/fineweb_7168_bpe.model"
      ;;
    *)
      die "Unsupported variant '$variant'. Expected one of: sp1024, sp8192, sp7680, sp7168."
      ;;
  esac
  RESOLVED_VOCAB_SIZE="$(variant_vocab_size "$variant")"
}

preset_metadata() {
  python3 - "$REPO_DIR" "$1" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from research.frontier_registry import FRONTIER_PRESETS
from research.presets import PRESETS

all_presets = {**PRESETS, **FRONTIER_PRESETS}
name = sys.argv[2]
if name not in all_presets:
    raise SystemExit(f"unknown preset:{name}")
preset = all_presets[name]
print(
    "\t".join(
        [
            preset.name,
            preset.lane,
            preset.target,
            preset.env.get("DATA_PATH", ""),
            preset.env.get("TOKENIZER_PATH", ""),
            preset.env.get("VOCAB_SIZE", ""),
        ]
    )
)
PY
}

variant_from_preset() {
  local preset_name="$1"
  local _name lane target data_path tokenizer_path vocab_size
  IFS=$'\t' read -r _name lane target data_path tokenizer_path vocab_size <<< "$(preset_metadata "$preset_name")"
  case "$vocab_size" in
    1024) printf 'sp1024\n' ;;
    8192) printf 'sp8192\n' ;;
    7680) printf 'sp7680\n' ;;
    7168) printf 'sp7168\n' ;;
    *)
      die "Cannot infer variant for preset '$preset_name' from VOCAB_SIZE='$vocab_size'."
      ;;
  esac
}

assert_stable_preset() {
  local preset_name="$1"
  local _name lane target data_path tokenizer_path vocab_size
  IFS=$'\t' read -r _name lane target data_path tokenizer_path vocab_size <<< "$(preset_metadata "$preset_name")"
  [ "$target" = "cuda" ] || die "Preset '$preset_name' targets '$target', but this RunPod flow is for frontier CUDA presets."
  case "$lane" in
    stable) ;;
    challenger)
      die "Preset '$preset_name' is challenger-only. Use research/run.py directly with --allow-challenger instead of this stable RunPod wrapper."
      ;;
    research_only)
      die "Preset '$preset_name' is research_only and is intentionally excluded from this submission-facing RunPod wrapper."
      ;;
    *)
      die "Preset '$preset_name' has unsupported lane '$lane'."
      ;;
  esac
}

assert_preset_variant_compatible() {
  local preset_name="$1"
  local variant="$2"
  local _name lane target data_path tokenizer_path vocab_size
  local expected_dataset expected_tokenizer
  resolve_variant_paths "$variant"
  expected_dataset="$(basename "$RESOLVED_DATA_PATH")"
  expected_tokenizer="$(basename "$RESOLVED_TOKENIZER_PATH")"
  IFS=$'\t' read -r _name lane target data_path tokenizer_path vocab_size <<< "$(preset_metadata "$preset_name")"
  [ "$(basename "$data_path")" = "$expected_dataset" ] || die "Preset '$preset_name' expects dataset $(basename "$data_path"), but variant '$variant' resolves to $expected_dataset."
  [ "$(basename "$tokenizer_path")" = "$expected_tokenizer" ] || die "Preset '$preset_name' expects tokenizer $(basename "$tokenizer_path"), but variant '$variant' resolves to $expected_tokenizer."
  [ "$vocab_size" = "$RESOLVED_VOCAB_SIZE" ] || die "Preset '$preset_name' expects VOCAB_SIZE=$vocab_size, but variant '$variant' resolves to $RESOLVED_VOCAB_SIZE."
}

resolve_effective_variant() {
  local stage="$1"
  local inferred_variant=""
  if [ "$VARIANT_EXPLICIT" -eq 1 ]; then
    RESOLVED_VARIANT="$VARIANT"
    return
  fi
  case "$stage" in
    prep)
      inferred_variant="$(variant_from_preset "$TRAIN_PRESET")"
      ;;
    smoke)
      inferred_variant="$(variant_from_preset "$SMOKE_PRESET")"
      ;;
    train)
      inferred_variant="$(variant_from_preset "$TRAIN_PRESET")"
      ;;
    all)
      local smoke_variant train_variant
      smoke_variant="$(variant_from_preset "$SMOKE_PRESET")"
      train_variant="$(variant_from_preset "$TRAIN_PRESET")"
      [ "$smoke_variant" = "$train_variant" ] || die "smoke preset '$SMOKE_PRESET' resolves to $smoke_variant but train preset '$TRAIN_PRESET' resolves to $train_variant. Pass an explicit compatible --variant or align the presets."
      inferred_variant="$train_variant"
      ;;
    *)
      die "Unsupported stage '$stage' for variant resolution."
      ;;
  esac
  RESOLVED_VARIANT="$VARIANT"
  if [ "$inferred_variant" != "$VARIANT" ]; then
    log "No explicit --variant provided; using preset-compatible variant '$inferred_variant' instead of default '$VARIANT'."
    RESOLVED_VARIANT="$inferred_variant"
  fi
}

ensure_repo() {
  log "STAGE repo: syncing $REPO_URL @ $BRANCH into $REPO_DIR"
  mkdir -p "$(dirname "$REPO_DIR")"
  if [ ! -e "$REPO_DIR" ]; then
    git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR"
  elif [ ! -d "$REPO_DIR/.git" ]; then
    die "Repo dir '$REPO_DIR' exists but is not a git checkout."
  fi
  (
    cd "$REPO_DIR"
    git remote set-url origin "$REPO_URL"
    git fetch origin --prune
    git checkout -B "$BRANCH" "origin/$BRANCH"
    git reset --hard "origin/$BRANCH"
  )
  export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
}

cleanup_transient_state() {
  log "STAGE cleanup: removing stale results, selected variant data, and transient caches"
  rm -rf "$REPO_DIR/research/results/runs"
  mkdir -p "$REPO_DIR/research/results/runs"
  rm -f "$REPO_DIR/research/results/index.jsonl" "$REPO_DIR/research/results/index.csv"
  if [ -d "$REPO_DIR/research_only/results" ]; then
    find "$REPO_DIR/research_only/results" -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +
  fi
  rm -rf "$RESOLVED_DATA_PATH"
  rm -f "$RESOLVED_TOKENIZER_PATH" "${RESOLVED_TOKENIZER_PATH%.model}.vocab"
  rm -f \
    "$REPO_DIR/data/manifest.json" \
    "$REPO_DIR/data/tokenizer_config.export.json" \
    "$REPO_DIR/data/docs_selected.jsonl" \
    "$REPO_DIR/data/docs_selected.source_manifest.json"
  rm -rf /tmp/pip-cache /tmp/pip-tmp
  mkdir -p /tmp/pip-cache /tmp/pip-tmp
}

ensure_python_deps() {
  log "STAGE deps: running approved cloud installer"
  (
    cd "$REPO_DIR"
    PYTHON=python3 ./scripts/install_cloud.sh || true
  )
  if ! python_import_available sentencepiece || ! python_import_available flash_attn; then
    log "STAGE deps: installing missing sentencepiece / flash-attn"
    mkdir -p /tmp/pip-tmp /tmp/pip-cache
    TMPDIR=/tmp/pip-tmp PIP_CACHE_DIR=/tmp/pip-cache python3 -m pip install sentencepiece flash-attn --no-build-isolation
  fi
  if ! python_import_available sentencepiece; then
    die "sentencepiece still does not import after install attempt."
  fi
  if ! python_import_available flash_attn; then
    die "flash_attn still does not import after install attempt."
  fi
}

wrapper_docs_bootstrap_command() {
  printf 'bash scripts/runpod_frontier.sh %s --variant %s --train-shards %s --allow-docs-bootstrap' "$STAGE" "$1" "$2"
}

direct_docs_bootstrap_command() {
  printf 'python3 data/download_hf_docs_and_tokenize.py --output-root ./data --tokenizer-config ./data/tokenizer_specs.json --variant %s --max-train-shards %s' "$1" "$2"
}

report_cached_variant_missing() {
  local variant="$1"
  local train_shards="$2"
  die "$(cat <<EOF
Requested variant: $variant
Cached manifest dataset_present=$CACHED_DATASET_PRESENT tokenizer_present=$CACHED_TOKENIZER_PRESENT
Manifest source: repo=$CACHED_REPO_ID root=$CACHED_REMOTE_ROOT_PREFIX path=$CACHED_MANIFEST_PATH
Prep stopped because the requested cached pretokenized artifacts are not published.
Ephemeral RunPod prep defaults to published cached artifacts only; docs bootstrap is disabled unless --allow-docs-bootstrap is set.
Manual wrapper command:
  $(wrapper_docs_bootstrap_command "$variant" "$train_shards")
Direct bootstrap command:
  $(direct_docs_bootstrap_command "$variant" "$train_shards")
Operator notes:
  - export HF_TOKEN=... before any large Hugging Face transfer
  - docs bootstrap is heavyweight and may require substantial free disk
  - publish the pretokenized dataset/tokenizer pair for fast pod startup
EOF
)"
}

docs_bootstrap_preflight() {
  local variant="$1"
  local free_bytes free_gib min_gib warn_gib
  free_bytes="$(free_disk_bytes "$REPO_DIR/data")"
  free_gib="$(format_bytes_gib "$free_bytes")"
  min_gib="$(format_bytes_gib "$DOCS_BOOTSTRAP_MIN_FREE_BYTES")"
  warn_gib="$(format_bytes_gib "$DOCS_BOOTSTRAP_WARN_FREE_BYTES")"
  if [ "$free_bytes" -lt "$DOCS_BOOTSTRAP_MIN_FREE_BYTES" ]; then
    die "$(cat <<EOF
Requested variant: $variant
Docs bootstrap was explicitly enabled, but prep stopped before download because free disk is too low.
Available free disk near $REPO_DIR/data: $free_gib
Required minimum for docs bootstrap: $min_gib
Docs bootstrap may transfer and reconstruct a very large docs_selected.jsonl blob before tokenization/export.
Manual bootstrap command:
  $(direct_docs_bootstrap_command "$variant" "$TRAIN_SHARDS")
Recommendation:
  - use published cached artifacts for ephemeral pods whenever possible
  - publish the pretokenized dataset/tokenizer pair for fast pod startup
EOF
)"
  fi
  if [ "$free_bytes" -lt "$DOCS_BOOTSTRAP_WARN_FREE_BYTES" ]; then
    log "WARNING: docs bootstrap free disk is only $free_gib; large Hugging Face reconstruction may be fragile. Conservative warning threshold: $warn_gib."
  fi
  if [ -z "${HF_TOKEN:-}" ]; then
    log "WARNING: HF_TOKEN is not set. Large Hugging Face transfers may be rate-limited or fail during docs bootstrap."
  fi
}

inspect_cached_variant_manifest() {
  local variant="$1"
  local summary_tsv
  summary_tsv="$(
    cd "$REPO_DIR"
    python3 data/cached_challenge_fineweb.py --variant "$variant" --check-only --json | \
      python3 -c 'import json, sys; summary = json.load(sys.stdin); print("\t".join([
summary.get("dataset_name", ""),
"1" if summary.get("dataset_present") else "0",
summary.get("tokenizer_name", "") or "",
"1" if summary.get("tokenizer_present") else "0",
summary.get("manifest_path", "") or "",
summary.get("repo_id", "") or "",
summary.get("remote_root_prefix", "") or "",
]))'
  )"
  IFS=$'\t' read -r CACHED_DATASET_NAME CACHED_DATASET_PRESENT CACHED_TOKENIZER_NAME CACHED_TOKENIZER_PRESENT CACHED_MANIFEST_PATH CACHED_REPO_ID CACHED_REMOTE_ROOT_PREFIX <<< "$summary_tsv"
  log "STAGE manifest: variant=$variant dataset_present=$CACHED_DATASET_PRESENT tokenizer_present=$CACHED_TOKENIZER_PRESENT manifest=$CACHED_MANIFEST_PATH"
}

fetch_variant_data() {
  local variant="$1"
  local train_shards="$2"
  log "STAGE data: fetching cached FineWeb artifacts for variant=$variant train_shards=$train_shards"
  if ! (
    cd "$REPO_DIR"
    python3 data/cached_challenge_fineweb.py --variant "$variant" --train-shards "$train_shards"
  ); then
    die "Failed to fetch cached artifacts for variant '$variant'. It may not be published in the cached manifest yet; publish it or rebuild it before running this stage."
  fi
}

bootstrap_variant_data() {
  local variant="$1"
  local train_shards="$2"
  docs_bootstrap_preflight "$variant"
  log "STAGE bootstrap: cached manifest is missing $CACHED_DATASET_NAME, rebuilding $variant from published docs_selected.jsonl"
  if ! (
    cd "$REPO_DIR"
    python3 data/download_hf_docs_and_tokenize.py \
      --output-root "$REPO_DIR/data" \
      --tokenizer-config "$REPO_DIR/data/tokenizer_specs.json" \
      --variant "$variant" \
      --max-train-shards "$train_shards"
  ); then
    die "Failed to bootstrap variant '$variant' from published docs in repo '$CACHED_REPO_ID' under '$CACHED_REMOTE_ROOT_PREFIX'. Retry manually with: python3 data/download_hf_docs_and_tokenize.py --output-root ./data --tokenizer-config ./data/tokenizer_specs.json --variant $variant --max-train-shards $train_shards. If docs_selected.jsonl is not published there, publish docs_selected/source_manifest or publish the pretokenized dataset."
  fi
}

ensure_variant_data() {
  local variant="$1"
  local train_shards="$2"
  inspect_cached_variant_manifest "$variant"
  if [ "$CACHED_DATASET_PRESENT" = "1" ] && [ "$CACHED_TOKENIZER_PRESENT" = "1" ]; then
    fetch_variant_data "$variant" "$train_shards"
    VARIANT_SOURCE="cached_manifest"
  else
    if [ "$ALLOW_DOCS_BOOTSTRAP" -ne 1 ]; then
      report_cached_variant_missing "$variant" "$train_shards"
    fi
    bootstrap_variant_data "$variant" "$train_shards"
    VARIANT_SOURCE="rebuild_from_docs"
  fi
  [ -d "$RESOLVED_DATA_PATH" ] || die "Expected dataset directory '$RESOLVED_DATA_PATH' was not created for variant '$variant'."
  [ -f "$RESOLVED_TOKENIZER_PATH" ] || die "Expected tokenizer file '$RESOLVED_TOKENIZER_PATH' was not created for variant '$variant'."
}

run_env_check() {
  log "STAGE env: checking CUDA / FlashAttention readiness"
  if [ "$ALLOW_MISSING_FLASH_ATTN" -eq 1 ]; then
    log "WARNING: --allow-missing-flash-attn is set. This is only intended for dry local or CPU edge cases."
    (
      cd "$REPO_DIR"
      python3 scripts/check_frontier_env.py --allow-missing-flash-attn
    )
  else
    (
      cd "$REPO_DIR"
      python3 scripts/check_frontier_env.py
    )
  fi
  log "ENV OK"
}

run_data_check() {
  local min_train_shards="$1"
  log "STAGE data: validating dataset/tokenizer pair"
  (
    cd "$REPO_DIR"
    python3 scripts/check_data.py \
      --data-path "$RESOLVED_DATA_PATH" \
      --tokenizer-path "$RESOLVED_TOKENIZER_PATH" \
      --min-train-shards "$min_train_shards" \
      --seq-len 2048
  )
  log "DATA OK"
}

print_ready_summary() {
  log "READY: repo synced and stable frontier prep completed"
  log "READY: variant=$RESOLVED_VARIANT source=$VARIANT_SOURCE dataset=$RESOLVED_DATA_PATH tokenizer=$RESOLVED_TOKENIZER_PATH train_shards=$TRAIN_SHARDS"
}

run_dry_run() {
  local preset="$1"
  local scale="$2"
  local run_name="$3"
  local nproc="$4"
  local gpu_profile="$5"
  log "STAGE dry-run: preset=$preset scale=$scale run_name=$run_name"
  (
    cd "$REPO_DIR"
    python3 research/run.py \
      --preset "$preset" \
      --scale "$scale" \
      --run-name "$run_name" \
      --seed "$SEED" \
      --nproc-per-node "$nproc" \
      --gpu-profile "$gpu_profile" \
      --set "DATA_PATH=$RESOLVED_DATA_PATH" \
      --set "TOKENIZER_PATH=$RESOLVED_TOKENIZER_PATH" \
      --set "VOCAB_SIZE=$RESOLVED_VOCAB_SIZE" \
      --dry-run \
      --skip-checks
  )
  log "DRY RUN OK"
}

run_preflight_only() {
  log "STAGE preflight: train preset preflight-only launch validation"
  (
    cd "$REPO_DIR"
    python3 research/run.py \
      --preset "$TRAIN_PRESET" \
      --scale submit_safe \
      --run-name "$TRAIN_RUN_NAME" \
      --seed "$SEED" \
      --nproc-per-node 8 \
      --gpu-profile "$TRAIN_GPU_PROFILE" \
      --set "DATA_PATH=$RESOLVED_DATA_PATH" \
      --set "TOKENIZER_PATH=$RESOLVED_TOKENIZER_PATH" \
      --set "VOCAB_SIZE=$RESOLVED_VOCAB_SIZE" \
      --preflight-only
  )
  log "PREFLIGHT OK"
}

run_smoke() {
  log "SMOKE START: preset=$SMOKE_PRESET run_name=$SMOKE_RUN_NAME"
  (
    cd "$REPO_DIR"
    python3 research/run.py \
      --preset "$SMOKE_PRESET" \
      --scale smoke \
      --run-name "$SMOKE_RUN_NAME" \
      --seed "$SEED" \
      --nproc-per-node 1 \
      --gpu-profile "$SMOKE_GPU_PROFILE" \
      --set "DATA_PATH=$RESOLVED_DATA_PATH" \
      --set "TOKENIZER_PATH=$RESOLVED_TOKENIZER_PATH" \
      --set "VOCAB_SIZE=$RESOLVED_VOCAB_SIZE"
  )
  log "SMOKE OK"
}

run_train() {
  local run_slug
  run_slug="$(slugify_name "$TRAIN_RUN_NAME")"
  log "TRAIN START: preset=$TRAIN_PRESET run_name=$TRAIN_RUN_NAME seed=$SEED"
  log "TRAIN START: expected run directory pattern $REPO_DIR/research/results/runs/*_${run_slug}"
  log "TRAIN START: expected logs/results under <run_dir>/${run_slug}.txt, launcher.log, result.json, run_summary.json"
  (
    cd "$REPO_DIR"
    python3 research/run.py \
      --preset "$TRAIN_PRESET" \
      --scale submit_safe \
      --run-name "$TRAIN_RUN_NAME" \
      --seed "$SEED" \
      --nproc-per-node 8 \
      --gpu-profile "$TRAIN_GPU_PROFILE" \
      --set "DATA_PATH=$RESOLVED_DATA_PATH" \
      --set "TOKENIZER_PATH=$RESOLVED_TOKENIZER_PATH" \
      --set "VOCAB_SIZE=$RESOLVED_VOCAB_SIZE"
  )
}

prepare_once() {
  local prep_preset="$1"
  local variant="$2"
  if [ "$PREP_DONE" -eq 1 ] && [ "$PREP_PRESET" = "$prep_preset" ] && [ "$PREP_VARIANT" = "$variant" ]; then
    return
  fi
  assert_stable_preset "$prep_preset"
  assert_preset_variant_compatible "$prep_preset" "$variant"
  resolve_variant_paths "$variant"
  cleanup_transient_state
  ensure_python_deps
  run_env_check
  ensure_variant_data "$variant" "$TRAIN_SHARDS"
  run_data_check "$TRAIN_SHARDS"
  PREP_DONE=1
  PREP_PRESET="$prep_preset"
  PREP_VARIANT="$variant"
  print_ready_summary
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    prep|smoke|train|all)
      [ -z "$STAGE" ] || die "Stage already set to '$STAGE'; got unexpected positional argument '$1'."
      STAGE="$1"
      shift
      ;;
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --variant)
      VARIANT="$2"
      VARIANT_EXPLICIT=1
      shift 2
      ;;
    --train-shards)
      TRAIN_SHARDS="$2"
      shift 2
      ;;
    --smoke-preset)
      SMOKE_PRESET="$2"
      shift 2
      ;;
    --train-preset)
      TRAIN_PRESET="$2"
      shift 2
      ;;
    --smoke-run-name)
      SMOKE_RUN_NAME="$2"
      SMOKE_RUN_NAME_EXPLICIT=1
      shift 2
      ;;
    --train-run-name)
      TRAIN_RUN_NAME="$2"
      TRAIN_RUN_NAME_EXPLICIT=1
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --smoke-gpu-profile)
      SMOKE_GPU_PROFILE="$2"
      shift 2
      ;;
    --train-gpu-profile)
      TRAIN_GPU_PROFILE="$2"
      shift 2
      ;;
    --skip-smoke)
      SKIP_SMOKE=1
      shift
      ;;
    --allow-docs-bootstrap)
      ALLOW_DOCS_BOOTSTRAP=1
      shift
      ;;
    --allow-missing-flash-attn)
      ALLOW_MISSING_FLASH_ATTN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[ -n "$STAGE" ] || {
  usage >&2
  exit 1
}

need_cmd bash
need_cmd git
need_cmd python3

REPO_DIR="$(abs_path "$REPO_DIR")"
[[ "$TRAIN_SHARDS" =~ ^[0-9]+$ ]] || die "--train-shards must be a non-negative integer."
[ "$TRAIN_SHARDS" -ge 1 ] || die "--train-shards must be at least 1 for smoke or train launches."

ensure_repo
resolve_effective_variant "$STAGE"
resolve_variant_paths "$RESOLVED_VARIANT"

if [ "$SMOKE_RUN_NAME_EXPLICIT" -eq 0 ] && [ "$SMOKE_RUN_NAME" = "sp1024_mainline_smoke" ] && [ "$RESOLVED_VARIANT" != "sp1024" ]; then
  SMOKE_RUN_NAME="${RESOLVED_VARIANT}_mainline_smoke"
fi
if [ "$TRAIN_RUN_NAME_EXPLICIT" -eq 0 ] && [ "$TRAIN_RUN_NAME" = "sp8192_mainline_submit" ] && [ "$RESOLVED_VARIANT" != "sp8192" ]; then
  TRAIN_RUN_NAME="${RESOLVED_VARIANT}_mainline_submit"
fi

log "Config: stage=$STAGE repo_dir=$REPO_DIR branch=$BRANCH variant=$RESOLVED_VARIANT train_shards=$TRAIN_SHARDS seed=$SEED"
log "Config: smoke_preset=$SMOKE_PRESET train_preset=$TRAIN_PRESET skip_smoke=$SKIP_SMOKE allow_docs_bootstrap=$ALLOW_DOCS_BOOTSTRAP"
log "Config: data_path=$RESOLVED_DATA_PATH tokenizer_path=$RESOLVED_TOKENIZER_PATH"

case "$STAGE" in
  prep)
    prepare_once "$TRAIN_PRESET" "$RESOLVED_VARIANT"
    run_dry_run "$TRAIN_PRESET" "submit_safe" "$TRAIN_RUN_NAME" "8" "$TRAIN_GPU_PROFILE"
    ;;
  smoke)
    prepare_once "$SMOKE_PRESET" "$RESOLVED_VARIANT"
    run_dry_run "$SMOKE_PRESET" "smoke" "$SMOKE_RUN_NAME" "1" "$SMOKE_GPU_PROFILE"
    run_smoke
    ;;
  train)
    prepare_once "$TRAIN_PRESET" "$RESOLVED_VARIANT"
    if [ "$SKIP_SMOKE" -eq 0 ]; then
      assert_stable_preset "$SMOKE_PRESET"
      assert_preset_variant_compatible "$SMOKE_PRESET" "$RESOLVED_VARIANT"
      run_dry_run "$SMOKE_PRESET" "smoke" "$SMOKE_RUN_NAME" "1" "$SMOKE_GPU_PROFILE"
      run_smoke
    fi
    run_dry_run "$TRAIN_PRESET" "submit_safe" "$TRAIN_RUN_NAME" "8" "$TRAIN_GPU_PROFILE"
    run_preflight_only
    run_train
    ;;
  all)
    prepare_once "$TRAIN_PRESET" "$RESOLVED_VARIANT"
    if [ "$SKIP_SMOKE" -eq 0 ]; then
      assert_stable_preset "$SMOKE_PRESET"
      assert_preset_variant_compatible "$SMOKE_PRESET" "$RESOLVED_VARIANT"
      run_dry_run "$SMOKE_PRESET" "smoke" "$SMOKE_RUN_NAME" "1" "$SMOKE_GPU_PROFILE"
      run_smoke
    fi
    run_dry_run "$TRAIN_PRESET" "submit_safe" "$TRAIN_RUN_NAME" "8" "$TRAIN_GPU_PROFILE"
    run_preflight_only
    run_train
    ;;
  *)
    die "Unsupported stage '$STAGE'."
    ;;
esac
