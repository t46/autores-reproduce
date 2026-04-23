"""Stage 3: Build execution environment for the paper's code.

Analyzes code dependencies, creates an isolated virtual environment
using uv, and installs all required packages. Handles common failures
like version conflicts and missing packages.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any


class EnvBuilder:
    """Builds isolated environments for experiment execution."""

    def __init__(self, output_dir: Path, verbose: bool = False):
        self.output_dir = output_dir
        self.verbose = verbose

    def build(self, code_info: dict[str, Any]) -> dict[str, Any]:
        """Build an environment for the given code.

        Args:
            code_info: Dict from CodeFinder with code path and metadata.

        Returns:
            Dict with environment path, python executable path, and metadata.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        code_path = Path(code_info.get("path", ""))

        if not code_path.exists():
            return {
                "success": False,
                "error": f"Code path does not exist: {code_path}",
            }

        # Detect dependencies
        deps = self._detect_dependencies(code_path)
        self._log(f"Detected dependencies: {deps}")

        # Detect framework
        framework = self._detect_framework(deps)
        self._log(f"Detected framework: {framework}")

        if not deps and framework == "unknown":
            self._log(
                "WARNING: Zero dependencies detected and framework is unknown. "
                "The code may be a placeholder or lack proper dependency declarations."
            )

        # Create virtual environment
        venv_path = self.output_dir / "venv"
        try:
            self._create_venv(venv_path)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create virtual environment: {e}",
            }

        # Install dependencies
        python_path = venv_path / "bin" / "python"

        # Detect whether to use `uv pip` or plain `pip`
        self._use_uv_pip = self._has_uv()

        try:
            self._install_dependencies(python_path, deps, code_path)
        except Exception as e:
            # Try minimal install as fallback
            self._log(f"Full install failed: {e}. Trying minimal install...")
            try:
                self._install_minimal(python_path, framework)
            except Exception as e2:
                return {
                    "success": False,
                    "error": f"Dependency installation failed: {e}. Minimal also failed: {e2}",
                }

        return {
            "success": True,
            "message": f"Environment built with {framework} framework",
            "venv_path": str(venv_path),
            "python_path": str(python_path),
            "framework": framework,
            "dependencies": deps,
        }

    def _detect_dependencies(self, code_path: Path) -> list[str]:
        """Detect dependencies from various configuration files."""
        deps = []

        # Check requirements.txt
        req_file = code_path / "requirements.txt"
        if req_file.exists():
            self._log("Found requirements.txt")
            content = req_file.read_text()
            for line in content.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    deps.append(line)

        # Check setup.py
        setup_file = code_path / "setup.py"
        if setup_file.exists():
            self._log("Found setup.py")
            content = setup_file.read_text()
            # Basic extraction of install_requires
            import re

            match = re.search(
                r"install_requires\s*=\s*\[(.*?)\]", content, re.DOTALL
            )
            if match:
                requires = match.group(1)
                for req in re.findall(r"['\"]([^'\"]+)['\"]", requires):
                    if req not in deps:
                        deps.append(req)

        # Check pyproject.toml
        pyproject_file = code_path / "pyproject.toml"
        if pyproject_file.exists():
            self._log("Found pyproject.toml")
            content = pyproject_file.read_text()
            # Basic TOML parsing for dependencies
            import re

            match = re.search(
                r"dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL
            )
            if match:
                dep_section = match.group(1)
                for dep in re.findall(r"['\"]([^'\"]+)['\"]", dep_section):
                    if dep not in deps:
                        deps.append(dep)

        # If no dependency files found, scan imports
        if not deps:
            self._log("No dependency files found, scanning imports...")
            deps = self._scan_imports(code_path)

        return deps

    def _scan_imports(self, code_path: Path) -> list[str]:
        """Scan Python files for imports and infer dependencies."""
        import re

        known_stdlib = {
            "os",
            "sys",
            "re",
            "json",
            "math",
            "time",
            "datetime",
            "pathlib",
            "collections",
            "itertools",
            "functools",
            "typing",
            "argparse",
            "logging",
            "subprocess",
            "shutil",
            "copy",
            "random",
            "abc",
            "dataclasses",
            "enum",
            "io",
            "glob",
            "pickle",
            "hashlib",
            "warnings",
            "contextlib",
            "multiprocessing",
            "threading",
            "queue",
            "tempfile",
            "csv",
            "struct",
        }

        imports = set()
        for py_file in code_path.rglob("*.py"):
            try:
                content = py_file.read_text()
                # Match 'import X' and 'from X import ...'
                for match in re.finditer(
                    r"^(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)", content, re.MULTILINE
                ):
                    pkg = match.group(1)
                    if pkg not in known_stdlib:
                        imports.add(pkg)
            except (UnicodeDecodeError, PermissionError):
                continue

        # Map common import names to pip package names
        import_to_package = {
            "torch": "torch",
            "torchvision": "torchvision",
            "tensorflow": "tensorflow",
            "tf": "tensorflow",
            "jax": "jax",
            "flax": "flax",
            "numpy": "numpy",
            "np": "numpy",
            "pandas": "pandas",
            "pd": "pandas",
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "transformers": "transformers",
            "datasets": "datasets",
            "scipy": "scipy",
            "matplotlib": "matplotlib",
            "tqdm": "tqdm",
            "yaml": "pyyaml",
            "wandb": "wandb",
            "tensorboard": "tensorboard",
            "einops": "einops",
            "timm": "timm",
        }

        deps = []
        for imp in imports:
            pkg_name = import_to_package.get(imp, imp)
            deps.append(pkg_name)

        return sorted(set(deps))

    def _detect_framework(self, deps: list[str]) -> str:
        """Detect the primary ML framework from dependencies."""
        dep_lower = [d.lower().split(">=")[0].split("==")[0].strip() for d in deps]

        if "torch" in dep_lower or "pytorch" in dep_lower:
            return "pytorch"
        elif "tensorflow" in dep_lower or "tf" in dep_lower:
            return "tensorflow"
        elif "jax" in dep_lower:
            return "jax"
        else:
            return "unknown"

    def _create_venv(self, venv_path: Path):
        """Create a virtual environment using uv or fallback to venv."""
        self._log(f"Creating virtual environment at: {venv_path}")

        # Try uv first
        try:
            result = subprocess.run(
                ["uv", "venv", str(venv_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                self._log("Created venv with uv")
                return
        except FileNotFoundError:
            pass

        # Fallback to stdlib venv
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"venv creation failed: {result.stderr}")
        self._log("Created venv with stdlib venv")

    @staticmethod
    def _has_uv() -> bool:
        """Check if `uv` is available on PATH."""
        try:
            subprocess.run(
                ["uv", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return True
        except FileNotFoundError:
            return False

    def _pip_install_cmd(self, python_path: Path) -> list[str]:
        """Return the base install command, preferring `uv pip` over plain pip."""
        if self._use_uv_pip:
            return ["uv", "pip", "install", "--python", str(python_path)]
        else:
            pip_path = python_path.parent / "pip"
            return [str(pip_path), "install"]

    def _install_dependencies(
        self, python_path: Path, deps: list[str], code_path: Path
    ):
        """Install dependencies into the virtual environment."""
        if not deps:
            self._log("No dependencies to install")
            return

        base_cmd = self._pip_install_cmd(python_path)

        # First try installing from requirements.txt if it exists
        req_file = code_path / "requirements.txt"
        if req_file.exists():
            self._log(f"Installing from requirements.txt (using {'uv pip' if self._use_uv_pip else 'pip'})")
            result = subprocess.run(
                base_cmd + ["-r", str(req_file)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                return
            self._log(f"requirements.txt install failed: {result.stderr[:500]}")

        # Fall back to installing deps individually
        self._log(f"Installing {len(deps)} dependencies individually")
        failed = []
        for dep in deps:
            result = subprocess.run(
                base_cmd + [dep],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                failed.append(dep)
                self._log(f"  Failed to install: {dep}")

        if len(failed) == len(deps):
            raise RuntimeError(f"All dependency installs failed: {failed}")
        elif failed:
            self._log(f"Warning: {len(failed)} deps failed to install: {failed}")

    def _install_minimal(self, python_path: Path, framework: str):
        """Install minimal dependencies based on detected framework."""
        minimal_deps = {
            "pytorch": ["torch", "torchvision", "numpy"],
            "tensorflow": ["tensorflow", "numpy"],
            "jax": ["jax", "jaxlib", "numpy"],
            "unknown": ["numpy"],
        }

        deps = minimal_deps.get(framework, ["numpy"])
        self._log(f"Installing minimal deps: {deps}")

        base_cmd = self._pip_install_cmd(python_path)
        result = subprocess.run(
            base_cmd + deps,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Minimal install failed: {result.stderr[:500]}")

    def _log(self, message: str):
        """Log if verbose mode is on."""
        if self.verbose:
            print(f"[EnvBuilder] {message}")
