from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


class RunPodFrontierScriptTest(unittest.TestCase):
    def test_shell_scripts_parse(self) -> None:
        for rel_path in (
            "scripts/runpod_frontier_common.sh",
            "scripts/runpod_frontier.sh",
            "scripts/install_cloud.sh",
        ):
            subprocess.run(
                ["bash", "-n", str(ROOT / rel_path)],
                check=True,
            )

    def test_dependency_bootstrap_checks_real_imports_and_installs_combined_command(self) -> None:
        script = (ROOT / "scripts/runpod_frontier.sh").read_text(encoding="utf-8")
        self.assertIn("python_import_available sentencepiece", script)
        self.assertIn("python_import_available flash_attn", script)
        self.assertIn(
            "TMPDIR=/tmp/pip-tmp PIP_CACHE_DIR=/tmp/pip-cache python3 -m pip install sentencepiece flash-attn --no-build-isolation",
            script,
        )
        self.assertIn('die "sentencepiece still does not import after install attempt."', script)
        self.assertIn('die "flash_attn still does not import after install attempt."', script)

    def test_prepare_once_runs_env_gate_after_dependency_bootstrap(self) -> None:
        script = (ROOT / "scripts/runpod_frontier.sh").read_text(encoding="utf-8")
        match = re.search(r"prepare_once\(\) \{\n(?P<body>.*?)\n\}", script, re.DOTALL)
        self.assertIsNotNone(match)
        body = match.group("body")
        self.assertLess(body.index("ensure_python_deps"), body.index("run_env_check"))

    def test_docs_bootstrap_is_explicit_opt_in(self) -> None:
        script = (ROOT / "scripts/runpod_frontier.sh").read_text(encoding="utf-8")
        self.assertIn("--allow-docs-bootstrap", script)
        match = re.search(r"ensure_variant_data\(\) \{\n(?P<body>.*?)\n\}", script, re.DOTALL)
        self.assertIsNotNone(match)
        body = match.group("body")
        self.assertIn('if [ "$ALLOW_DOCS_BOOTSTRAP" -ne 1 ]; then', body)
        self.assertIn("report_cached_variant_missing", body)

    def test_docs_bootstrap_preflight_mentions_disk_and_hf_token(self) -> None:
        script = (ROOT / "scripts/runpod_frontier.sh").read_text(encoding="utf-8")
        self.assertIn("DOCS_BOOTSTRAP_MIN_FREE_BYTES", script)
        self.assertIn("DOCS_BOOTSTRAP_WARN_FREE_BYTES", script)
        self.assertIn("free_disk_bytes", script)
        self.assertIn("HF_TOKEN", script)
        self.assertIn("publish the pretokenized dataset/tokenizer pair for fast pod startup", script)

    def test_explicit_sp1024_variant_selects_sp1024_compatible_default_presets(self) -> None:
        script = (ROOT / "scripts/runpod_frontier.sh").read_text(encoding="utf-8")
        match = re.search(r"(usage\(\) \{.*)\nwhile \[ \"\$#\" -gt 0 \]; do", script, re.DOTALL)
        self.assertIsNotNone(match)
        harness = f"""
set -euo pipefail
SCRIPT_DIR={str(ROOT / "scripts")!r}
source "$SCRIPT_DIR/runpod_frontier_common.sh"
{match.group(1)}
REPO_DIR={str(ROOT)!r}
VARIANT=sp1024
VARIANT_EXPLICIT=1
SMOKE_PRESET_EXPLICIT=0
TRAIN_PRESET_EXPLICIT=0
RESOLVED_VARIANT=sp1024
resolve_variant_default_presets
printf '%s\\n%s\\n' "$SMOKE_PRESET" "$TRAIN_PRESET"
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".sh") as f:
            f.write(harness)
            f.flush()
            result = subprocess.run(
                ["bash", f.name],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertIn("smoke_preset=baseline", result.stdout)
        self.assertIn("train_preset=sota_plus_ppm_dirichlet_submit", result.stdout)
        resolved = result.stdout.strip().splitlines()[-2:]
        self.assertEqual(resolved, ["baseline", "sota_plus_ppm_dirichlet_submit"])
        self.assertFalse(any(preset.startswith("sp8192_") for preset in resolved))


if __name__ == "__main__":
    unittest.main()
