from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
TOKENIZERS_DIR = ROOT / "tokenizers"


def _hf_hub_download(*args, **kwargs):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency availability varies by environment
        raise RuntimeError("huggingface_hub is required for cached FineWeb downloads") from exc
    return hf_hub_download(*args, **kwargs)

def dataset_dir_for_variant(name: str) -> str:
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(f"unsupported variant {name!r}; expected byte260 or sp<VOCAB_SIZE>")


def expected_tokenizer_name_for_variant(name: str) -> str | None:
    if name == "byte260":
        return "pure_byte_260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"sp_bpe_{name[2:]}"
    return None


def local_path_for_remote(relative_path: str) -> Path:
    remote_path = Path(relative_path)
    if REMOTE_ROOT_PREFIX and remote_path.parts[:1] == (REMOTE_ROOT_PREFIX,):
        remote_path = remote_path.relative_to(REMOTE_ROOT_PREFIX)
    if remote_path.parts[:1] == ("datasets",):
        return DATASETS_DIR.joinpath(*remote_path.parts[1:])
    if remote_path.parts[:1] == ("tokenizers",):
        return TOKENIZERS_DIR.joinpath(*remote_path.parts[1:])
    return ROOT / remote_path


def get(relative_path: str) -> None:
    destination = local_path_for_remote(relative_path)
    if destination.exists():
        return
    if destination.is_symlink():
        destination.unlink()

    remote_path = Path(relative_path)
    cached_path = Path(
        _hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path.name,
            subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
            repo_type="dataset",
        )
    )
    # HF cache entries may be snapshot symlinks. Resolve to the underlying blob so we
    # always materialize a real file in data/, not a broken relative symlink.
    cached_source = cached_path.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(cached_source, destination)
    except OSError:
        shutil.copy2(cached_source, destination)


def manifest_path() -> Path:
    return local_path_for_remote(f"{REMOTE_ROOT_PREFIX}/manifest.json")


def load_manifest(*, skip_manifest_download: bool) -> dict:
    path = manifest_path()
    if not path.is_file():
        if skip_manifest_download:
            raise FileNotFoundError(
                f"manifest.json is required for manifest-driven shard counts but is not present locally at {path}"
            )
        get(f"{REMOTE_ROOT_PREFIX}/manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_paths_for_tokenizer(tokenizer_entry: dict) -> list[str]:
    artifacts = []
    for key in ("model_path", "vocab_path", "path"):
        value = tokenizer_entry.get(key)
        if value:
            artifacts.append(str(value))
    if not artifacts:
        raise ValueError(f"tokenizer entry is missing downloadable artifacts: {tokenizer_entry}")
    return artifacts


def manifest_variant_summary(manifest: dict, variant: str) -> dict[str, object]:
    dataset_name = dataset_dir_for_variant(variant)
    expected_tokenizer_name = expected_tokenizer_name_for_variant(variant)
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_name), None)
    tokenizer_name = dataset_entry.get("tokenizer_name") if dataset_entry is not None else expected_tokenizer_name
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    stats = (dataset_entry or {}).get("stats") or {}
    artifact_paths: list[str] = []
    if tokenizer_entry is not None:
        artifact_paths = artifact_paths_for_tokenizer(tokenizer_entry)
    return {
        "variant": variant,
        "dataset_name": dataset_name,
        "dataset_present": dataset_entry is not None,
        "tokenizer_name": tokenizer_name,
        "expected_tokenizer_name": expected_tokenizer_name,
        "tokenizer_present": tokenizer_entry is not None,
        "max_train_shards": None if dataset_entry is None else int(stats.get("files_train", 0)),
        "val_shards": None if dataset_entry is None else int(stats.get("files_val", 0)),
        "artifact_paths": artifact_paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download challenge FineWeb shards from Hugging Face")
    parser.add_argument(
        "train_shards_positional",
        nargs="?",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        default=80,
        help="Number of training shards to download for the selected variant. Defaults to 80.",
    )
    parser.add_argument(
        "--variant",
        default="sp1024",
        help="Tokenizer family to download, for example sp1024, sp4096, or byte260.",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip downloading manifest.json.",
    )
    parser.add_argument(
        "--with-docs",
        action="store_true",
        help="Also download docs_selected.jsonl and its sidecar for tokenizer retraining or dataset re-export.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Inspect manifest availability for the requested variant without downloading shards.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = dataset_dir_for_variant(args.variant)
    train_shards = args.train_shards_positional if args.train_shards_positional is not None else args.train_shards
    if train_shards < 0:
        raise ValueError("train_shards must be non-negative")

    manifest = load_manifest(skip_manifest_download=args.skip_manifest)
    summary = manifest_variant_summary(manifest, args.variant)
    summary["repo_id"] = REPO_ID
    summary["remote_root_prefix"] = REMOTE_ROOT_PREFIX
    summary["manifest_path"] = str(manifest_path())
    if args.check_only:
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"variant: {summary['variant']}")
            print(f"dataset_name: {summary['dataset_name']}")
            print(f"dataset_present: {summary['dataset_present']}")
            print(f"tokenizer_name: {summary['tokenizer_name']}")
            print(f"tokenizer_present: {summary['tokenizer_present']}")
            print(f"repo_id: {summary['repo_id']}")
            print(f"remote_root_prefix: {summary['remote_root_prefix']}")
            print(f"manifest_path: {summary['manifest_path']}")
            if summary.get("max_train_shards") is not None:
                print(f"max_train_shards: {summary['max_train_shards']}")
            if summary.get("val_shards") is not None:
                print(f"val_shards: {summary['val_shards']}")
            for artifact_path in summary.get("artifact_paths") or []:
                print(f"artifact_path: {artifact_path}")
        return

    if not summary["dataset_present"]:
        raise ValueError(
            f"dataset {dataset_dir} not found in {REMOTE_ROOT_PREFIX}/manifest.json from {REPO_ID}"
        )
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir), None)
    assert dataset_entry is not None
    max_train_shards = int((dataset_entry.get("stats") or {}).get("files_train"))
    val_shards = int((dataset_entry.get("stats") or {}).get("files_val"))
    if train_shards > max_train_shards:
        raise ValueError(
            f"{args.variant} only has {max_train_shards} training shards on {REPO_ID}, requested {train_shards}"
        )
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    if tokenizer_entry is None:
        raise ValueError(f"tokenizer {tokenizer_name} not found in {REMOTE_ROOT_PREFIX}/manifest.json from {REPO_ID}")

    if args.with_docs:
        get(f"{REMOTE_ROOT_PREFIX}/docs_selected.jsonl")
        get(f"{REMOTE_ROOT_PREFIX}/docs_selected.source_manifest.json")

    dataset_prefix = f"{REMOTE_ROOT_PREFIX}/datasets/{dataset_dir}"
    for i in range(val_shards):
        get(f"{dataset_prefix}/fineweb_val_{i:06d}.bin")
    for i in range(train_shards):
        get(f"{dataset_prefix}/fineweb_train_{i:06d}.bin")

    for artifact_path in artifact_paths_for_tokenizer(tokenizer_entry):
        get(f"{REMOTE_ROOT_PREFIX}/{artifact_path}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:  # pragma: no cover - shell pipeline behavior
        sys.exit(0)
