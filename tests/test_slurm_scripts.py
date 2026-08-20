from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPO_ROOT / "scripts" / "slurm"
SBATCH_SCRIPTS = (
    "run_single.sh",
    "run_distributed.sh",
    "run_scaling.sh",
    "run_plot.sh",
    "run_webui.sh",
)


class SlurmScriptContractTest(unittest.TestCase):
    def test_all_shell_scripts_are_valid_bash(self) -> None:
        scripts = sorted(str(path) for path in SLURM_DIR.glob("*.sh"))
        completed = subprocess.run(
            ["bash", "-n", *scripts],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_sbatch_scripts_use_submit_directory(self) -> None:
        for name in SBATCH_SCRIPTS:
            with self.subTest(script=name):
                content = (SLURM_DIR / name).read_text()
                self.assertIn("SLURM_SUBMIT_DIR", content)
                self.assertIn("BASH_SOURCE[0]", content)

    def test_environment_setup_preserves_lockfile(self) -> None:
        content = (SLURM_DIR / "setup_env.sh").read_text()
        self.assertIn("npm ci", content)
        self.assertNotIn("rm -rf node_modules package-lock.json", content)
        self.assertNotIn("--omit=optional", content)
        self.assertIn('PYTORCH_VERSION="${PYTORCH_VERSION:-2.7.1}"', content)

    def test_monitor_supports_combined_and_legacy_logs(self) -> None:
        content = (SLURM_DIR / "monitor.sh").read_text()
        self.assertIn('.log', content)
        self.assertIn('.out', content)

    def test_multi_node_submission_allocates_one_task_per_node(self) -> None:
        content = (SLURM_DIR / "run_gpu_experiments.sh").read_text()
        self.assertIn("--ntasks-per-node=1", content)
        self.assertNotIn("eval ${CMD}", content)
        scaling_content = (SLURM_DIR / "run_scaling.sh").read_text()
        self.assertIn("#SBATCH --ntasks-per-node=1", scaling_content)
        self.assertNotIn("#SBATCH --ntasks=1", scaling_content)
        self.assertIn("WORKER_TMPDIR", scaling_content)

    def test_long_running_jobs_exit_after_signals(self) -> None:
        for name in ("run_webui.sh", "run_distributed.sh", "run_scaling.sh"):
            with self.subTest(script=name):
                content = (SLURM_DIR / name).read_text()
                self.assertIn("trap cleanup EXIT", content)
                self.assertIn("trap 'exit 143' TERM", content)


if __name__ == "__main__":
    unittest.main()
