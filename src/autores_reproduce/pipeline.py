"""Main pipeline orchestrator for paper reproduction.

Coordinates all stages of the reproduction pipeline:
1. Fetch paper from arXiv
2. Find or generate code
3. Build environment
4. Execute experiments
5. Verify results against paper claims
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .code_finder import CodeFinder
from .env_builder import EnvBuilder
from .executor import Executor
from .paper_fetcher import PaperFetcher
from .verifier import Verifier


class ReproductionPipeline:
    """Orchestrates the end-to-end paper reproduction process."""

    def __init__(
        self,
        arxiv_url: str,
        output_dir: Path,
        timeout: int = 3600,
        use_gpu: bool = True,
        verbose: bool = False,
    ):
        self.arxiv_url = arxiv_url
        self.output_dir = output_dir.resolve()
        self.timeout = timeout
        self.use_gpu = use_gpu
        self.verbose = verbose

        self.stages: list[dict[str, Any]] = []
        self.paper_info: dict[str, Any] = {}
        self.code_info: dict[str, Any] = {}
        self.env_info: dict[str, Any] = {}
        self.execution_info: dict[str, Any] = {}
        self.verification_info: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        """Run the full reproduction pipeline.

        Returns a comprehensive report dict. Continues through stages
        even if individual stages fail partially.
        """
        self._log("Starting reproduction pipeline")
        start_time = time.time()

        # Stage 1: Fetch paper
        self.paper_info = self._run_stage(
            "Paper Fetching",
            self._fetch_paper,
        )

        # Stage 2: Find/generate code (requires paper info)
        if self.paper_info.get("success"):
            self.code_info = self._run_stage(
                "Code Finding",
                self._find_code,
            )
        else:
            self.code_info = {"success": False, "error": "Skipped: paper fetch failed"}
            self.stages.append(
                {
                    "name": "Code Finding",
                    "success": False,
                    "message": "Skipped due to paper fetch failure",
                    "duration": 0,
                }
            )

        # Stage 3: Build environment (requires code)
        if self.code_info.get("success"):
            self.env_info = self._run_stage(
                "Environment Building",
                self._build_env,
            )
        else:
            self.env_info = {"success": False, "error": "Skipped: code finding failed"}
            self.stages.append(
                {
                    "name": "Environment Building",
                    "success": False,
                    "message": "Skipped due to code finding failure",
                    "duration": 0,
                }
            )

        # Stage 4: Execute experiments (requires environment)
        if self.env_info.get("success"):
            self.execution_info = self._run_stage(
                "Experiment Execution",
                self._execute,
            )
        else:
            self.execution_info = {
                "success": False,
                "error": "Skipped: env build failed",
            }
            self.stages.append(
                {
                    "name": "Experiment Execution",
                    "success": False,
                    "message": "Skipped due to environment build failure",
                    "duration": 0,
                }
            )

        # Stage 5: Verify results (can run with partial results)
        if self.execution_info.get("success"):
            self.verification_info = self._run_stage(
                "Result Verification",
                self._verify,
            )
        else:
            self.verification_info = {
                "success": False,
                "error": "Skipped: execution failed",
            }
            self.stages.append(
                {
                    "name": "Result Verification",
                    "success": False,
                    "message": "Skipped due to execution failure",
                    "duration": 0,
                }
            )

        total_duration = time.time() - start_time
        return self._compile_report(total_duration)

    def _run_stage(self, name: str, func) -> dict[str, Any]:
        """Run a pipeline stage with timing and error handling."""
        self._log(f"Stage: {name}")
        start = time.time()
        try:
            result = func()
            duration = time.time() - start
            stage_entry = {
                "name": name,
                "success": result.get("success", False),
                "message": result.get("message", result.get("error", "")),
                "duration": round(duration, 2),
            }
            if not result.get("success") and "error" in result:
                stage_entry["error"] = result["error"]
            self.stages.append(stage_entry)
            return result
        except Exception as e:
            duration = time.time() - start
            self.stages.append(
                {
                    "name": name,
                    "success": False,
                    "error": str(e),
                    "message": f"Stage failed with exception: {type(e).__name__}",
                    "duration": round(duration, 2),
                }
            )
            return {"success": False, "error": str(e)}

    def _fetch_paper(self) -> dict[str, Any]:
        """Stage 1: Fetch and parse paper."""
        fetcher = PaperFetcher(self.output_dir / "paper", verbose=self.verbose)
        return fetcher.fetch(self.arxiv_url)

    def _find_code(self) -> dict[str, Any]:
        """Stage 2: Find or generate code."""
        finder = CodeFinder(self.output_dir / "code", verbose=self.verbose)
        return finder.find(self.paper_info)

    def _build_env(self) -> dict[str, Any]:
        """Stage 3: Build execution environment."""
        builder = EnvBuilder(self.output_dir / "env", verbose=self.verbose)
        return builder.build(self.code_info)

    def _execute(self) -> dict[str, Any]:
        """Stage 4: Run experiments."""
        executor = Executor(
            self.output_dir / "execution",
            timeout=self.timeout,
            use_gpu=self.use_gpu,
            verbose=self.verbose,
        )
        return executor.run(self.code_info, self.env_info)

    def _verify(self) -> dict[str, Any]:
        """Stage 5: Verify results against paper claims."""
        verifier = Verifier(verbose=self.verbose)
        return verifier.verify(self.paper_info, self.execution_info)

    def _compile_report(self, total_duration: float) -> dict[str, Any]:
        """Compile the final reproduction report."""
        all_succeeded = all(s.get("success", False) for s in self.stages)
        any_succeeded = any(s.get("success", False) for s in self.stages)

        if all_succeeded:
            status = "completed"
        elif any_succeeded:
            status = "partial"
        else:
            status = "failed"

        return {
            "arxiv_url": self.arxiv_url,
            "paper_title": self.paper_info.get("title", "Unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "reproduction_score": self.verification_info.get("score"),
            "total_duration": round(total_duration, 2),
            "stages": self.stages,
            "claims": self.verification_info.get("claims", []),
            "paper_info": {
                k: v
                for k, v in self.paper_info.items()
                if k not in ("success", "message")
            },
            "code_info": {
                "source": self.code_info.get("source"),
                "path": str(self.code_info.get("path", "")),
            },
            "config": {
                "timeout": self.timeout,
                "use_gpu": self.use_gpu,
            },
        }

    def generate_failure_report(self, error: str) -> dict[str, Any]:
        """Generate a report when the pipeline fails catastrophically."""
        return {
            "arxiv_url": self.arxiv_url,
            "paper_title": self.paper_info.get("title", "Unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "reproduction_score": None,
            "total_duration": 0,
            "stages": self.stages,
            "claims": [],
            "error": error,
            "paper_info": {},
            "code_info": {},
            "config": {
                "timeout": self.timeout,
                "use_gpu": self.use_gpu,
            },
        }

    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[Pipeline] {message}")
