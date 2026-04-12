from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(rel_path: str, module_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


cached_challenge_fineweb = _load_module("data/cached_challenge_fineweb.py", "cached_challenge_fineweb")
download_hf_docs_and_tokenize = _load_module("data/download_hf_docs_and_tokenize.py", "download_hf_docs_and_tokenize")


class CachedVariantManifestTest(unittest.TestCase):
    def test_manifest_summary_reports_missing_dataset_but_present_tokenizer(self) -> None:
        manifest = {
            "tokenizers": [
                {
                    "name": "sp_bpe_8192",
                    "model_path": "tokenizers/fineweb_8192_bpe.model",
                    "vocab_path": "tokenizers/fineweb_8192_bpe.vocab",
                }
            ],
            "datasets": [],
        }
        summary = cached_challenge_fineweb.manifest_variant_summary(manifest, "sp8192")
        self.assertEqual(summary["dataset_name"], "fineweb10B_sp8192")
        self.assertFalse(summary["dataset_present"])
        self.assertEqual(summary["tokenizer_name"], "sp_bpe_8192")
        self.assertTrue(summary["tokenizer_present"])
        self.assertIn("tokenizers/fineweb_8192_bpe.model", summary["artifact_paths"])


class DocsRetokenizeBootstrapTest(unittest.TestCase):
    def test_filter_specs_for_variant_selects_only_requested_family(self) -> None:
        specs = [
            {"name": "sp_bpe_1024", "dataset_suffix": "sp1024", "vocab_size": 1024},
            {"name": "sp_bpe_8192", "dataset_suffix": "sp8192", "vocab_size": 8192},
            {"name": "sp_bpe_7680", "dataset_suffix": "sp7680", "vocab_size": 7680},
        ]
        filtered = download_hf_docs_and_tokenize.filter_specs_for_variant(specs, "sp8192")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["name"], "sp_bpe_8192")

    def test_export_shards_can_cap_training_shards_for_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            docs_path = td_path / "docs_selected.jsonl"
            docs = [{"text": "alpha beta gamma delta epsilon zeta eta theta"} for _ in range(6)]
            docs_path.write_text("".join(json.dumps(item) + "\n" for item in docs), encoding="utf-8")
            tok = download_hf_docs_and_tokenize.build_pure_byte_tokenizer(
                spec={"name": "pure_byte_260", "dataset_suffix": "byte260"},
                docs_jsonl=docs_path,
                tokenizers_dir=td_path / "tokenizers",
            )
            output_dir = td_path / "datasets" / "fineweb10B_byte260"
            stats = download_hf_docs_and_tokenize.export_shards(
                docs_path,
                tok,
                output_dir,
                num_val_docs=1,
                shard_size=48,
                docs_total=len(docs),
                max_train_shards=1,
            )
            self.assertEqual(stats["files_train"], 1)
            self.assertEqual(stats["train_shards_requested"], 1)
            self.assertTrue(stats["train_subset_only"])
            self.assertEqual(len(list(output_dir.glob("fineweb_train_*.bin"))), 1)
            self.assertGreaterEqual(len(list(output_dir.glob("fineweb_val_*.bin"))), 1)


if __name__ == "__main__":
    unittest.main()
