"""Tests for the reproduction pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autores_reproduce.paper_fetcher import PaperFetcher
from autores_reproduce.code_finder import CodeFinder
from autores_reproduce.env_builder import EnvBuilder
from autores_reproduce.executor import Executor
from autores_reproduce.verifier import Verifier
from autores_reproduce.pipeline import ReproductionPipeline


class TestPaperFetcher:
    """Tests for paper fetching and parsing."""

    def test_extract_arxiv_id_abs_url(self):
        fetcher = PaperFetcher(Path("/tmp/test"))
        assert fetcher._extract_arxiv_id("https://arxiv.org/abs/2301.12345") == "2301.12345"

    def test_extract_arxiv_id_pdf_url(self):
        fetcher = PaperFetcher(Path("/tmp/test"))
        assert fetcher._extract_arxiv_id("https://arxiv.org/pdf/2301.12345.pdf") == "2301.12345"

    def test_extract_arxiv_id_with_version(self):
        fetcher = PaperFetcher(Path("/tmp/test"))
        assert fetcher._extract_arxiv_id("https://arxiv.org/abs/2301.12345v2") == "2301.12345v2"

    def test_extract_arxiv_id_bare_id(self):
        fetcher = PaperFetcher(Path("/tmp/test"))
        assert fetcher._extract_arxiv_id("2301.12345") == "2301.12345"

    def test_extract_arxiv_id_invalid(self):
        fetcher = PaperFetcher(Path("/tmp/test"))
        assert fetcher._extract_arxiv_id("not a valid url") is None


class TestCodeFinder:
    """Tests for code finding and empty repo detection."""

    def test_has_python_files_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "main.py").write_text("print('hello')")
            assert CodeFinder._has_python_files(Path(tmpdir)) is True

    def test_has_python_files_false_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert CodeFinder._has_python_files(Path(tmpdir)) is False

    def test_has_python_files_false_only_readme(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "README.md").write_text("# Placeholder repo")
            (Path(tmpdir) / "LICENSE").write_text("MIT")
            assert CodeFinder._has_python_files(Path(tmpdir)) is False

    def test_has_python_files_nested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "src" / "models"
            subdir.mkdir(parents=True)
            (subdir / "model.py").write_text("class Model: pass")
            assert CodeFinder._has_python_files(Path(tmpdir)) is True

    def test_try_clone_repo_rejects_empty_repo(self):
        """Verify that _try_clone_repo returns None for repos with no .py files."""
        finder = CodeFinder(Path(tempfile.mkdtemp()), verbose=True)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            # Create the target dir with only a README (simulating empty repo)
            target_dir = finder.output_dir / "repo"
            target_dir.mkdir(parents=True)
            (target_dir / "README.md").write_text("# Empty repo")

            result = finder._try_clone_repo("https://github.com/user/repo")
            assert result is None
            # The directory should have been cleaned up
            assert not target_dir.exists()


class TestEnvBuilder:
    """Tests for environment building."""

    def test_detect_framework_pytorch(self):
        builder = EnvBuilder(Path("/tmp/test"))
        deps = ["torch>=2.0", "torchvision", "numpy"]
        assert builder._detect_framework(deps) == "pytorch"

    def test_detect_framework_tensorflow(self):
        builder = EnvBuilder(Path("/tmp/test"))
        deps = ["tensorflow>=2.10", "numpy"]
        assert builder._detect_framework(deps) == "tensorflow"

    def test_detect_framework_jax(self):
        builder = EnvBuilder(Path("/tmp/test"))
        deps = ["jax", "jaxlib", "flax"]
        assert builder._detect_framework(deps) == "jax"

    def test_detect_framework_unknown(self):
        builder = EnvBuilder(Path("/tmp/test"))
        deps = ["numpy", "pandas"]
        assert builder._detect_framework(deps) == "unknown"

    def test_detect_dependencies_from_requirements(self):
        builder = EnvBuilder(Path("/tmp/test"))
        with tempfile.TemporaryDirectory() as tmpdir:
            req_file = Path(tmpdir) / "requirements.txt"
            req_file.write_text("torch>=2.0\nnumpy\n# comment\nPillow\n")
            deps = builder._detect_dependencies(Path(tmpdir))
            assert "torch>=2.0" in deps
            assert "numpy" in deps
            assert "Pillow" in deps


class TestExecutor:
    """Tests for experiment execution."""

    def test_extract_metrics_accuracy(self):
        executor = Executor(Path("/tmp/test"))
        output = "Epoch 10: loss: 0.234, accuracy: 0.956"
        metrics = executor._extract_metrics(output)
        assert "accuracy" in metrics
        assert abs(metrics["accuracy"] - 0.956) < 0.001

    def test_extract_metrics_multiple(self):
        executor = Executor(Path("/tmp/test"))
        output = """
        Training complete.
        Test loss: 0.123
        Test accuracy: 92.5%
        F1: 0.89
        """
        metrics = executor._extract_metrics(output)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "f1" in metrics

    def test_find_entry_point(self):
        executor = Executor(Path("/tmp/test"))
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "main.py").write_text("print('hello')")
            (Path(tmpdir) / "utils.py").write_text("# utils")
            entry = executor._find_entry_point(Path(tmpdir))
            assert entry is not None
            assert entry.name == "main.py"

    def test_find_entry_point_train(self):
        executor = Executor(Path("/tmp/test"))
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "train.py").write_text("print('train')")
            entry = executor._find_entry_point(Path(tmpdir))
            assert entry is not None
            assert entry.name == "train.py"

    def test_find_entry_point_returns_none_for_empty_dir(self):
        executor = Executor(Path("/tmp/test"))
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = executor._find_entry_point(Path(tmpdir))
            assert entry is None

    def test_run_no_entry_point_returns_error_message(self):
        """Ensure error message is propagated when no entry point is found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "code"
            code_path.mkdir()
            # Only a README, no Python files
            (code_path / "README.md").write_text("# Empty")

            executor = Executor(Path(tmpdir) / "out")
            result = executor.run(
                {"path": str(code_path)},
                {"python_path": "python"},
            )
            assert result["success"] is False
            assert "message" in result
            assert "entry point" in result["message"].lower()
            assert result["error"] == result["message"]

    def test_summarize_error_oom(self):
        executor = Executor(Path("/tmp/test"))
        stderr = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
        msg = executor._summarize_error(stderr, 1)
        assert "out of memory" in msg.lower() or "GPU" in msg


class TestVerifier:
    """Tests for result verification."""

    def test_verify_claim_match(self):
        verifier = Verifier(tolerance=0.05)
        claim = {"metric": "accuracy", "value": "0.95", "description": "Test acc"}
        metrics = {"accuracy": 0.94}
        result = verifier._verify_claim(claim, metrics, "")
        # 0.94 is within 5% of 0.95 (diff is ~1%)
        assert result["status"] == "verified"

    def test_verify_claim_fail(self):
        verifier = Verifier(tolerance=0.05)
        claim = {"metric": "accuracy", "value": "0.95", "description": "Test acc"}
        metrics = {"accuracy": 0.80}
        result = verifier._verify_claim(claim, metrics, "")
        assert result["status"] == "failed"

    def test_verify_claim_untested(self):
        verifier = Verifier(tolerance=0.05)
        claim = {"metric": "bleu", "value": "32.5", "description": "BLEU score"}
        metrics = {"accuracy": 0.95}  # No BLEU metric
        result = verifier._verify_claim(claim, metrics, "")
        assert result["status"] == "untested"

    def test_calculate_score(self):
        verifier = Verifier()
        claims = [
            {"status": "verified"},
            {"status": "verified"},
            {"status": "failed"},
            {"status": "untested"},
        ]
        score = verifier._calculate_score(claims)
        # (2 + 0.5*1) / 4 = 2.5/4 = 0.625
        assert abs(score - 0.625) < 0.001

    def test_calculate_score_all_verified(self):
        verifier = Verifier()
        claims = [{"status": "verified"}, {"status": "verified"}]
        assert verifier._calculate_score(claims) == 1.0

    def test_calculate_score_empty(self):
        verifier = Verifier()
        assert verifier._calculate_score([]) == 0.0


class TestPipeline:
    """Tests for the pipeline orchestrator."""

    def test_generate_failure_report(self):
        pipeline = ReproductionPipeline(
            arxiv_url="https://arxiv.org/abs/2301.12345",
            output_dir=Path("/tmp/test"),
        )
        report = pipeline.generate_failure_report("Something went wrong")
        assert report["status"] == "failed"
        assert report["error"] == "Something went wrong"
        assert report["arxiv_url"] == "https://arxiv.org/abs/2301.12345"

    def test_run_stage_propagates_error_to_stages(self):
        """When a stage returns success=False with error, the error appears in stages."""
        pipeline = ReproductionPipeline(
            arxiv_url="https://arxiv.org/abs/2301.12345",
            output_dir=Path("/tmp/test"),
        )

        def failing_stage():
            return {"success": False, "error": "No entry point found"}

        result = pipeline._run_stage("Test Stage", failing_stage)
        assert result["success"] is False
        assert len(pipeline.stages) == 1
        stage = pipeline.stages[0]
        assert stage["success"] is False
        assert "error" in stage
        assert stage["error"] == "No entry point found"
        # message should fall back to error when no message key is present
        assert stage["message"] == "No entry point found"
