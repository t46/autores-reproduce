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

            # 2026-04-30: success_strict — multi-signal で「実は失敗」を検出
            # returncode=0 でも以下のいずれかなら strict failure:
            #   - silent failure pattern が stderr/stdout に出ている
            #   - metric が 1 個も抽出できなかった (実行は終わったが何も測れていない)
            silent_fail_patterns = [
                "RuntimeError",
                "mat1 and mat2",
                "shape mismatch",
                "size mismatch",
                "dimension mismatch",
                "Traceback (most recent call last)",
                # 注: 大文字小文字を区別する。"NaN" "nan" は単なる loss 値表示の可能性があるので除外
            ]
            combined_output = (result.stderr or "") + "\n" + (result.stdout or "")
            silent_failure_hits = [p for p in silent_fail_patterns if p in combined_output]
            success_strict = (
                success
                and len(metrics) > 0
                and not silent_failure_hits
            )
            success_strict_reason = None
            if success and not success_strict:
                if not metrics:
                    success_strict_reason = "returncode=0 だが metric が 1 個も抽出できなかった"
                elif silent_failure_hits:
                    success_strict_reason = (
                        f"returncode=0 だが silent failure pattern を検出: {silent_failure_hits[:3]}"
                    )

            return {
                "success": success,
                "success_strict": success_strict,
                "success_strict_reason": success_strict_reason,
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

        # Common metric patterns (expanded 2026-04-30)
        # Number pattern includes scientific notation and negatives
        N = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        kv_sep = r"\s*[:=]\s*"
        patterns = [
            # Classification / regression
            (rf"(?:test|eval|val)?\s*(?:accuracy|acc){kv_sep}{N}%?", "accuracy"),
            (rf"(?:test|eval|val)?\s*(?:loss){kv_sep}{N}", "loss"),
            (rf"(?:f1|f1[._-]score|macro[._-]f1|micro[._-]f1){kv_sep}{N}", "f1"),
            (rf"(?:precision){kv_sep}{N}", "precision"),
            (rf"(?:recall){kv_sep}{N}", "recall"),
            (rf"(?:perplexity|ppl){kv_sep}{N}", "perplexity"),
            (rf"(?:mse|mean[._-]squared[._-]error){kv_sep}{N}", "mse"),
            (rf"(?:mae|mean[._-]absolute[._-]error){kv_sep}{N}", "mae"),
            (rf"(?:auc|auroc|auc[._-]roc){kv_sep}{N}", "auc"),
            (rf"(?:mean[._-]average[._-]precision|\bmap\b){kv_sep}{N}", "map"),
            # NLP generation
            (rf"(?:bleu(?:[._-]?\d)?|bleu[._-]score){kv_sep}{N}", "bleu"),
            (rf"(?:rouge[._-]?[12l]?|rouge[._-]?score){kv_sep}{N}", "rouge"),
            (rf"(?:meteor){kv_sep}{N}", "meteor"),
            (rf"(?:cider(?:[._-]?d)?){kv_sep}{N}", "cider"),
            (rf"(?:spice){kv_sep}{N}", "spice"),
            (rf"(?:bert[._-]?score){kv_sep}{N}", "bertscore"),
            (rf"(?:chrf(?:\+\+|[._-]?pp)?){kv_sep}{N}", "chrf"),
            (rf"(?:bleurt){kv_sep}{N}", "bleurt"),
            (rf"(?:comet[._-]?score|\bcomet\b){kv_sep}{N}", "comet"),
            (rf"(?:sari){kv_sep}{N}", "sari"),
            (rf"(?:mauve){kv_sep}{N}", "mauve"),
            # Speech / sequence
            (rf"(?:wer|word[._-]error[._-]rate){kv_sep}{N}", "wer"),
            (rf"(?:cer|character[._-]error[._-]rate){kv_sep}{N}", "cer"),
            # QA / reading
            (rf"(?:exact[._-]?match|\bem\b){kv_sep}{N}", "em"),
            # Code
            (rf"(?:pass\s*@?\s*1|pass[._-]?1){kv_sep}{N}", "pass@1"),
            (rf"(?:pass\s*@?\s*(?:k|10|100)){kv_sep}{N}", "pass@k"),
            # Image generation / quality
            (rf"(?:fid[._-]?50k?){kv_sep}{N}", "fid-50k"),
            (rf"(?:fid[._-]?score|frechet[._-]inception[._-]distance|\bfid\b){kv_sep}{N}", "fid"),
            (rf"(?:lpips){kv_sep}{N}", "lpips"),
            (rf"(?:ssim){kv_sep}{N}", "ssim"),
            (rf"(?:psnr){kv_sep}{N}", "psnr"),
            (rf"(?:inception[._-]?score|\bis[._-]?score\b){kv_sep}{N}", "is_score"),
            (rf"(?:kid[._-]?score|kernel[._-]inception[._-]distance|\bkid\b){kv_sep}{N}", "kid"),
            (rf"(?:clip[._-]?score){kv_sep}{N}", "clip_score"),
            # Density / generative
            (rf"(?:negative[._-]log[._-]likelihood|\bnll\b){kv_sep}{N}", "nll"),
            (rf"(?:elbo|evidence[._-]lower[._-]bound){kv_sep}{N}", "elbo"),
            (rf"(?:kl[._-]?divergence|\bkl\b|\bkld\b){kv_sep}{N}", "kl"),
            # Segmentation / detection
            (rf"(?:mean[._-]?iou|miou|\biou\b|intersection[._-]over[._-]union){kv_sep}{N}", "iou"),
            (rf"(?:dice[._-]?(?:score|coefficient)?){kv_sep}{N}", "dice"),
            # Top-k
            (rf"(?:top[._-]?1(?:[._-]?acc(?:uracy)?)?){kv_sep}{N}", "top1"),
            (rf"(?:top[._-]?5(?:[._-]?acc(?:uracy)?)?){kv_sep}{N}", "top5"),
            # Retrieval / ranking
            (rf"(?:mean[._-]reciprocal[._-]rank|\bmrr\b){kv_sep}{N}", "mrr"),
            (rf"(?:ndcg(?:@\d+)?){kv_sep}{N}", "ndcg"),
            (rf"(?:hits@\d+|recall@\d+|precision@\d+){kv_sep}{N}", "hits@k"),
            # Bookkeeping
            (rf"epoch{kv_sep}?{N}", "epochs_completed"),
        ]

        for pattern, metric_name in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
            if matches:
                try:
                    metrics[metric_name] = float(matches[-1])
                except (ValueError, TypeError):
                    pass

        # Additionally: parse markdown-style tables (`| FID | 1.13 |`)
        table_row = re.compile(
            rf"\|\s*([A-Za-z][\w@\-/.]*)\s*\|\s*{N}\s*\|",
            re.IGNORECASE,
        )
        for m in table_row.finditer(output):
            metric_name_raw = m.group(1).strip().lower().replace(" ", "")
            try:
                value = float(m.group(2))
            except (ValueError, TypeError):
                continue
            # Only keep if not already extracted by a more specific pattern
            if metric_name_raw and metric_name_raw not in metrics:
                metrics[metric_name_raw] = value

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
