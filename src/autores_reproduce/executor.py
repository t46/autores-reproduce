"""Stage 4: Execute experiments from the paper.

Identifies the entry point, configures for available hardware,
runs with timeout and resource limits, and captures all output.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Any


class Executor:
    """Executes paper experiments in the prepared environment."""

    def __init__(
        self,
        output_dir: Path,
        timeout: int = 3600,
        use_gpu: bool = True,
        verbose: bool = False,
    ):
        self.output_dir = output_dir
        self.timeout = timeout
        self.use_gpu = use_gpu
        self.verbose = verbose

    def run(
        self, code_info: dict[str, Any], env_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the paper's experiments.

        Args:
            code_info: Dict from CodeFinder with code path.
            env_info: Dict from EnvBuilder with environment details.

        Returns:
            Dict with execution results, stdout/stderr, and metrics.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        code_path = Path(code_info.get("path", ""))
        python_path = env_info.get("python_path", "python")

        if not code_path.exists():
            return {
                "success": False,
                "error": f"Code path does not exist: {code_path}",
            }

        # Find entry point
        entry_point = self._find_entry_point(code_path)
        if not entry_point:
            error_msg = (
                "Could not find entry point (main.py, train.py, run.py, etc.) "
                f"in {code_path}"
            )
            self._log(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": error_msg,
            }

        self._log(f"Entry point: {entry_point}")

        # Detect GPU availability
        gpu_available = self._detect_gpu() if self.use_gpu else False
        self._log(f"GPU available: {gpu_available}")

        # Build execution command
        cmd = self._build_command(python_path, entry_point, gpu_available)
        self._log(f"Execution command: {' '.join(cmd)}")

        # Set up environment variables
        env = self._build_env_vars(gpu_available)

        # Execute
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(code_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            duration = time.time() - start_time

            # Save outputs
            stdout_path = self.output_dir / "stdout.txt"
            stderr_path = self.output_dir / "stderr.txt"
            stdout_path.write_text(result.stdout)
            stderr_path.write_text(result.stderr)

            # Extract metrics from output
            metrics = self._extract_metrics(result.stdout + "\n" + result.stderr)

            # Collect output files
            output_files = self._collect_output_files(code_path)

            success = result.returncode == 0
            if not success:
                error_summary = self._summarize_error(result.stderr, result.returncode)
            else:
                error_summary = None

            return {
                "success": success,
                "message": f"Execution {'completed' if success else 'failed'} in {duration:.1f}s",
                "returncode": result.returncode,
                "duration": round(duration, 2),
                "stdout": result.stdout[-10000:],  # Last 10k chars
                "stderr": result.stderr[-5000:],  # Last 5k chars
                "metrics": metrics,
                "output_files": output_files,
                "error": error_summary,
                "gpu_used": gpu_available,
                "entry_point": str(entry_point),
            }

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "success": False,
                "error": f"Execution timed out after {self.timeout}s",
                "duration": round(duration, 2),
                "metrics": {},
                "output_files": [],
            }

    def _find_entry_point(self, code_path: Path) -> Path | None:
        """Find the main entry point for execution.

        Searches for common entry point patterns in order of priority.
        """
        candidates = [
            "main.py",
            "train.py",
            "run.py",
            "run_experiment.py",
            "run_experiments.py",
            "experiment.py",
            "train_model.py",
            "run_training.py",
        ]

        # Check root level first
        for name in candidates:
            path = code_path / name
            if path.exists():
                return path

        # Check common subdirectories
        for subdir in ["src", "scripts", "tools"]:
            for name in candidates:
                path = code_path / subdir / name
                if path.exists():
                    return path

        # Look for any Python file with if __name__ == "__main__"
        for py_file in sorted(code_path.rglob("*.py")):
            # Skip test files and __init__.py
            if "test" in py_file.name.lower() or py_file.name == "__init__.py":
                continue
            try:
                content = py_file.read_text()
                if '__name__' in content and '__main__' in content:
                    return py_file
            except (UnicodeDecodeError, PermissionError):
                continue

        # Check for shell scripts
        for name in ["run.sh", "train.sh", "run_experiment.sh"]:
            path = code_path / name
            if path.exists():
                return path

        return None

    def _detect_gpu(self) -> bool:
        """Detect if GPU (CUDA) is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _build_command(
        self, python_path: str, entry_point: Path, gpu_available: bool
    ) -> list[str]:
        """Build the execution command."""
        if entry_point.suffix == ".sh":
            return ["bash", str(entry_point)]

        cmd = [python_path, str(entry_point)]

        # Add common flags if GPU is not available
        if not gpu_available:
            # Try to force CPU usage (common patterns)
            cmd_str = entry_point.read_text() if entry_point.exists() else ""
            if "--device" in cmd_str or "device" in cmd_str:
                cmd.extend(["--device", "cpu"])
            elif "--no-cuda" in cmd_str:
                cmd.append("--no-cuda")

        return cmd

    def _build_env_vars(self, gpu_available: bool) -> dict[str, str]:
        """Build environment variables for execution."""
        env = os.environ.copy()

        if not gpu_available:
            env["CUDA_VISIBLE_DEVICES"] = ""

        # Common environment settings
        env["PYTHONUNBUFFERED"] = "1"

        return env

    def _extract_metrics(self, output: str) -> dict[str, Any]:
        """Extract numerical metrics from execution output.

        Looks for common patterns like:
            accuracy: 0.95
            loss: 0.234
            F1: 0.89
            BLEU: 32.5
        """
        import re

        metrics = {}

        # Common metric patterns
        patterns = [
            (r"(?:test|eval|val)?\s*(?:accuracy|acc)\s*[:=]\s*([\d.]+)%?", "accuracy"),
            (r"(?:test|eval|val)?\s*(?:loss)\s*[:=]\s*([\d.]+)", "loss"),
            (r"(?:f1|f1.score)\s*[:=]\s*([\d.]+)", "f1"),
            (r"(?:precision)\s*[:=]\s*([\d.]+)", "precision"),
            (r"(?:recall)\s*[:=]\s*([\d.]+)", "recall"),
            (r"(?:bleu)\s*[:=]\s*([\d.]+)", "bleu"),
            (r"(?:rouge.?1?)\s*[:=]\s*([\d.]+)", "rouge"),
            (r"(?:perplexity|ppl)\s*[:=]\s*([\d.]+)", "perplexity"),
            (r"(?:mse|mean.squared.error)\s*[:=]\s*([\d.]+)", "mse"),
            (r"(?:mae|mean.absolute.error)\s*[:=]\s*([\d.]+)", "mae"),
            (r"(?:auc|auroc)\s*[:=]\s*([\d.]+)", "auc"),
            (r"(?:map|mAP)\s*[:=]\s*([\d.]+)", "map"),
            (r"epoch\s*[:=]?\s*(\d+)", "epochs_completed"),
        ]

        for pattern, metric_name in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                # Take the last occurrence (final result)
                try:
                    metrics[metric_name] = float(matches[-1])
                except ValueError:
                    pass

        return metrics

    def _collect_output_files(self, code_path: Path) -> list[str]:
        """Collect output files generated during execution."""
        output_extensions = {
            ".pt",
            ".pth",
            ".h5",
            ".ckpt",
            ".json",
            ".csv",
            ".txt",
            ".log",
            ".png",
            ".pdf",
        }
        output_dirs = ["output", "outputs", "results", "logs", "checkpoints"]

        files = []
        for dirname in output_dirs:
            dir_path = code_path / dirname
            if dir_path.exists():
                for f in dir_path.rglob("*"):
                    if f.is_file() and f.suffix in output_extensions:
                        files.append(str(f.relative_to(code_path)))

        return files[:50]  # Limit to 50 files

    def _summarize_error(self, stderr: str, returncode: int) -> str:
        """Summarize execution errors into a readable message."""
        common_errors = {
            "CUDA out of memory": "GPU out of memory. Try reducing batch size or using CPU.",
            "ModuleNotFoundError": "Missing Python module. Environment may be incomplete.",
            "FileNotFoundError": "Missing file or dataset. May need to download data first.",
            "RuntimeError: CUDA": "CUDA error. GPU may not be properly configured.",
            "PermissionError": "Permission denied. Check file permissions.",
            "ConnectionError": "Network error. May need to download data/models.",
            "ValueError": "Value error in code. May be a configuration issue.",
        }

        for pattern, message in common_errors.items():
            if pattern in stderr:
                return f"{message} (exit code: {returncode})"

        # Return last meaningful line of stderr
        lines = [l.strip() for l in stderr.strip().split("\n") if l.strip()]
        if lines:
            return f"{lines[-1]} (exit code: {returncode})"

        return f"Execution failed with exit code: {returncode}"

    def _log(self, message: str):
        """Log if verbose mode is on."""
        if self.verbose:
            print(f"[Executor] {message}")
