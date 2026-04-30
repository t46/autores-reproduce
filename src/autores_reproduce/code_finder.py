"""Stage 2: Find official code or generate implementation.

Searches for official code repositories on GitHub by checking paper
references and searching by title/arXiv ID. If no official code is found,
uses Claude to generate an implementation from the paper description.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import requests
from anthropic import Anthropic


class CodeFinder:
    """Finds or generates code for a paper."""

    def __init__(self, output_dir: Path, verbose: bool = False):
        self.output_dir = output_dir
        self.verbose = verbose
        self.client = Anthropic()

    def find(self, paper_info: dict[str, Any]) -> dict[str, Any]:
        """Find official code or generate implementation.

        Args:
            paper_info: Dict from PaperFetcher with paper metadata and analysis.

        Returns:
            Dict with code path, source type (official/generated), and metadata.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Check for code references in paper
        code_refs = paper_info.get("code_references", [])
        if code_refs:
            self._log(f"Found {len(code_refs)} code references in paper")
            for ref in code_refs:
                result = self._try_clone_repo(ref)
                if result:
                    return {
                        "success": True,
                        "source": "official",
                        "message": f"Cloned official repository: {ref}",
                        "path": result,
                        "repo_url": ref,
                    }

        # Step 2: Search GitHub
        self._log("Searching GitHub for code...")
        github_result = self._search_github(paper_info)
        if github_result:
            return {
                "success": True,
                "source": "github_search",
                "message": f"Found code via GitHub search: {github_result['url']}",
                "path": github_result["path"],
                "repo_url": github_result["url"],
            }

        # Step 3: Generate code with Claude
        self._log("No official code found. Generating implementation with Claude...")
        gen_result = self._generate_code(paper_info)
        if gen_result:
            return {
                "success": True,
                "source": "generated",
                "message": "Generated implementation from paper description",
                "path": gen_result,
                "repo_url": None,
            }

        return {
            "success": False,
            "error": "Could not find or generate code for this paper",
        }

    def _try_clone_repo(self, url: str) -> Path | None:
        """Attempt to clone a repository from a URL."""
        if not url or not isinstance(url, str):
            return None

        # Normalize GitHub URLs
        github_match = re.search(
            r"github\.com/([^/]+/[^/\s]+)", url.rstrip("/").rstrip(".git")
        )
        if not github_match:
            return None

        repo_path = github_match.group(1)
        clone_url = f"https://github.com/{repo_path}.git"
        target_dir = self.output_dir / "repo"

        self._log(f"Attempting to clone: {clone_url}")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, str(target_dir)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                if not self._has_python_files(target_dir):
                    self._log(
                        "Clone successful but repo contains no .py files "
                        "(likely a placeholder). Discarding."
                    )
                    import shutil

                    shutil.rmtree(target_dir, ignore_errors=True)
                    return None
                self._log("Clone successful")
                return target_dir
            else:
                self._log(f"Clone failed: {result.stderr}")
                return None
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self._log(f"Clone error: {e}")
            return None

    def _search_github(self, paper_info: dict[str, Any]) -> dict[str, Any] | None:
        """Search GitHub for the paper's code repository."""
        title = paper_info.get("title", "")
        arxiv_id = paper_info.get("arxiv_id", "")

        queries = []
        if arxiv_id:
            queries.append(arxiv_id)
        if title:
            # Use first few significant words of the title
            words = [w for w in title.split() if len(w) > 3][:5]
            queries.append(" ".join(words))

        headers = {"Accept": "application/vnd.github.v3+json"}

        for query in queries:
            self._log(f"GitHub search query: {query}")
            try:
                response = requests.get(
                    "https://api.github.com/search/repositories",
                    params={"q": query, "sort": "stars", "per_page": 5},
                    headers=headers,
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])
                    if items:
                        # Take the top result
                        repo = items[0]
                        clone_url = repo["clone_url"]
                        target_dir = self.output_dir / "repo"

                        result = subprocess.run(
                            [
                                "git",
                                "clone",
                                "--depth",
                                "1",
                                clone_url,
                                str(target_dir),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                        if result.returncode == 0:
                            if not self._has_python_files(target_dir):
                                self._log(
                                    f"GitHub repo {repo['html_url']} contains "
                                    "no .py files. Skipping."
                                )
                                import shutil

                                shutil.rmtree(target_dir, ignore_errors=True)
                                continue
                            return {
                                "path": target_dir,
                                "url": repo["html_url"],
                            }
                elif response.status_code == 403:
                    self._log("GitHub API rate limit reached")
                    break
            except (requests.RequestException, subprocess.TimeoutExpired) as e:
                self._log(f"GitHub search error: {e}")
                continue

        return None

    def _generate_code(self, paper_info: dict[str, Any]) -> Path | None:
        """Generate implementation code using Claude."""
        gen_dir = self.output_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        title = paper_info.get("title", "Unknown")
        abstract = paper_info.get("abstract", "")
        architecture = paper_info.get("architecture", {})
        training = paper_info.get("training", {})
        datasets = paper_info.get("datasets", [])

        prompt = f"""Generate a complete, runnable Python implementation for the following ML paper.

Paper: {title}
Abstract: {abstract}

Architecture details: {json.dumps(architecture, indent=2) if isinstance(architecture, dict) else str(architecture)}
Training details: {json.dumps(training, indent=2) if isinstance(training, dict) else str(training)}
Datasets: {json.dumps(datasets, indent=2) if isinstance(datasets, list) else str(datasets)}

Requirements:
1. Create a self-contained implementation that can be run with `python main.py`
2. Use PyTorch as the framework
3. Include:
   - Model definition (model.py)
   - Training script (train.py)
   - Main entry point (main.py) that trains and evaluates
   - requirements.txt with dependencies
4. Use standard datasets from torchvision/HuggingFace when possible
5. Include reasonable default hyperparameters from the paper
6. Print metrics (loss, accuracy, etc.) during training
7. The code should actually run and produce results (even if simplified)

Format your response as a series of files, each preceded by a line like:
=== filename.py ===
followed by the file content.

Keep the implementation focused and practical. It's OK to simplify complex architectures
for reproducibility - the goal is to verify the paper's main claims."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16384,  # 2026-04-30: 8192 -> 16384 (cut-off 緩和)
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text
        except Exception as e:
            self._log(f"Claude API error: {e}")
            return None

        # Parse the response into files
        files_written = self._parse_and_write_files(response_text, gen_dir)

        if files_written:
            self._log(f"Generated {len(files_written)} files")
            return gen_dir
        else:
            self._log("Failed to parse generated code")
            return None

    def _parse_and_write_files(
        self, response_text: str, output_dir: Path
    ) -> list[str]:
        """Parse Claude's response into individual files."""
        files_written = []

        # Split on file markers like === filename.py ===
        parts = re.split(r"===\s*(.+?)\s*===", response_text)

        # parts[0] is preamble, then alternating (filename, content)
        for i in range(1, len(parts) - 1, 2):
            filename = parts[i].strip()
            content = parts[i + 1].strip()

            # Remove markdown code fences if present
            content = re.sub(r"^```\w*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

            filepath = output_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content)
            files_written.append(filename)
            self._log(f"  Written: {filename}")

        # Fallback: if no === markers, try to find code blocks
        if not files_written:
            code_blocks = re.findall(
                r"```(?:python)?\n(.*?)```", response_text, re.DOTALL
            )
            if code_blocks:
                # Write the largest block as main.py
                largest = max(code_blocks, key=len)
                main_path = output_dir / "main.py"
                main_path.write_text(largest)
                files_written.append("main.py")

                # Write requirements
                req_path = output_dir / "requirements.txt"
                req_path.write_text("torch\ntorchvision\nnumpy\n")
                files_written.append("requirements.txt")

        return files_written

    @staticmethod
    def _has_python_files(directory: Path) -> bool:
        """Check if a directory tree contains at least one .py file."""
        return any(directory.rglob("*.py"))

    def _log(self, message: str):
        """Log if verbose mode is on."""
        if self.verbose:
            print(f"[CodeFinder] {message}")
