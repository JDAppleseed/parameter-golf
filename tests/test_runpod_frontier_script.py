from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunPodFrontierScriptTest(unittest.TestCase):
    def test_shell_scripts_parse(self) -> None:
        for rel_path in (
            "scripts/runpod_frontier_common.sh",
            "scripts/runpod_frontier.sh",
        ):
            subprocess.run(
                ["bash", "-n", str(ROOT / rel_path)],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
