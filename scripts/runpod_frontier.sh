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
ALLOW_MISSING_FLASH_ATTN=0
STAGE=""
RESOLVED_VARIANT=""
RESOLVED_DATA_PATH=""
RESOLVED_TOKENIZER_PATH=""
RESOLVED_VOCAB_SIZE=""
PREP_DONE=0
PREP_PRESET=""
PREP_VARIANT=""

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

ensure_python_deps() {
  log "STAGE deps: running approved cloud installer"
  (
    cd "$REPO_DIR"
    PYTHON=python3 ./scripts/install_cloud.sh || true
  )
  if ! python3 -c "import sentencepiece" >/dev/null 2>&1; then
    log "Installing missing dependency: sentencepiece"
    TMPDIR=/tmp/pip-tmp PIP_CACHE_DIR=/tmp/pip-cache python3 -m pip install sentencepiece
  fi
  if ! python3 -c "import flash_attn_interface" >/dev/null 2>&1; then
    log "Installing missing dependency: flash-attn"
    TMPDIR=/tmp/pip-tmp PIP_CACHE_DIR=/tmp/pip-cache python3 -m pip install flash-attn --no-build-isolation
  fi
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
  ensure_python_deps
  run_env_check
  fetch_variant_data "$variant" "$TRAIN_SHARDS"
  run_data_check "$TRAIN_SHARDS"
  PREP_DONE=1
  PREP_PRESET="$prep_preset"
  PREP_VARIANT="$variant"
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
log "Config: smoke_preset=$SMOKE_PRESET train_preset=$TRAIN_PRESET skip_smoke=$SKIP_SMOKE"
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
