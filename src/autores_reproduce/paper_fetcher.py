"""Stage 1: Fetch paper from arXiv and extract key information.

Downloads the paper PDF, fetches metadata via the arXiv API, and uses
Claude to extract experimental claims, datasets, architecture details,
training procedures, and reported metrics.
"""

import re
from pathlib import Path
from typing import Any

import arxiv
from anthropic import Anthropic
from PyPDF2 import PdfReader

# 2026-04-30: pdfplumber を優先 extractor に追加。table 抽出が PyPDF2 より遥かに強い。
# import 失敗時は PyPDF2 fallback で動くように構造を組む。
try:
    import pdfplumber  # type: ignore
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False


class PaperFetcher:
    """Fetches and parses ML papers from arXiv."""

    def __init__(self, output_dir: Path, verbose: bool = False):
        self.output_dir = output_dir
        self.verbose = verbose
        self.client = Anthropic()

    def fetch(self, arxiv_url: str, paper_cache_dir: Path | str | None = None) -> dict[str, Any]:
        """Fetch paper metadata, PDF, and extract key information.

        Args:
            arxiv_url: URL to the arXiv paper (abs or pdf link).
            paper_cache_dir: Optional path to a paperbench-data style directory containing
                paper.pdf + paper.md (+ optional config.yaml). When provided, arXiv fetch is
                bypassed and metadata + PDF text are read locally. This avoids 429 rate limits
                during batch experiments. Added 2026-04-30 for pipeline-rethink batch.

        Returns:
            Dict with paper metadata, extracted claims, and analysis.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract arXiv ID from URL
        arxiv_id = self._extract_arxiv_id(arxiv_url)
        if not arxiv_id:
            return {
                "success": False,
                "error": f"Could not extract arXiv ID from URL: {arxiv_url}",
            }

        self._log(f"Fetching paper with arXiv ID: {arxiv_id}")

        if paper_cache_dir is not None:
            # Cached path: read locally, no arXiv API call
            try:
                metadata, pdf_path, paper_text = self._fetch_from_cache(arxiv_id, Path(paper_cache_dir))
                self._log(f"Loaded paper from cache: {paper_cache_dir}")
            except Exception as e:
                return {"success": False, "error": f"Failed to load cached paper: {e}"}
        else:
            # Fetch metadata via arXiv API
            try:
                metadata = self._fetch_metadata(arxiv_id)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to fetch arXiv metadata: {e}",
                }

            # Download PDF
            try:
                pdf_path = self._download_pdf(metadata)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to download PDF: {e}",
                    **metadata,
                }

            # Extract text from PDF
            try:
                paper_text = self._extract_pdf_text(pdf_path)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to extract PDF text: {e}",
                    **metadata,
                }

        # Use Claude to analyze the paper
        try:
            analysis = self._analyze_paper(paper_text, metadata)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to analyze paper with Claude: {e}",
                **metadata,
            }

        return {
            "success": True,
            "message": f"Successfully fetched and analyzed: {metadata['title']}",
            **metadata,
            **analysis,
            "pdf_path": str(pdf_path),
            # 2026-04-30: 50000 -> 200000 に拡張。Claude Sonnet 4 の context は十分。
            "paper_text": paper_text[:200000],
        }

    def _fetch_from_cache(self, arxiv_id: str, cache_dir: Path) -> tuple[dict, Path, str]:
        """Load paper.{pdf,md} from a paperbench-data style cache directory.

        Returns (metadata, pdf_path, paper_text) tuple. Bypasses arXiv API + download.
        """
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"cache_dir not found: {cache_dir}")
        pdf_src = cache_dir / "paper.pdf"
        md_src = cache_dir / "paper.md"
        if not pdf_src.exists():
            raise FileNotFoundError(f"paper.pdf missing in {cache_dir}")
        # Copy PDF into output_dir so downstream paths look the same as the online flow
        pdf_dest = self.output_dir / "paper.pdf"
        pdf_dest.write_bytes(pdf_src.read_bytes())

        # Title / abstract: prefer paper.md (first non-empty line as title, optional Abstract section)
        title = arxiv_id
        abstract = ""
        if md_src.exists():
            md_text = md_src.read_text(errors="replace")
            for line in md_text.splitlines():
                line = line.strip()
                if line:
                    # Strip leading "# " markdown
                    title = line.lstrip("#").strip() or arxiv_id
                    break
            # Best-effort abstract extraction
            m = re.search(r"(?:^|\n)#+\s*Abstract\s*\n+([^\n#]+(?:\n[^\n#]+)*)", md_text, re.IGNORECASE)
            if m:
                abstract = m.group(1).strip()[:2000]

        metadata = {
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": [],
            "published": "",
            "pdf_url": str(pdf_src),
        }

        # Use the existing pdfplumber-based extractor on the cached PDF
        paper_text = self._extract_pdf_text(pdf_dest)
        return metadata, pdf_dest, paper_text

    def _extract_arxiv_id(self, url: str) -> str | None:
        """Extract arXiv ID from various URL formats.

        Handles:
            https://arxiv.org/abs/2301.12345
            https://arxiv.org/pdf/2301.12345.pdf
            https://arxiv.org/abs/2301.12345v2
            2301.12345
        """
        patterns = [
            r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)",
            r"^(\d{4}\.\d{4,5}(?:v\d+)?)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _fetch_metadata(self, arxiv_id: str) -> dict[str, Any]:
        """Fetch paper metadata from arXiv API."""
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))

        if not results:
            raise ValueError(f"No paper found with ID: {arxiv_id}")

        paper = results[0]
        return {
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "authors": [str(a) for a in paper.authors],
            "abstract": paper.summary,
            "pdf_url": paper.pdf_url,
            "published": paper.published.isoformat() if paper.published else None,
            "categories": paper.categories,
        }

    def _download_pdf(self, metadata: dict[str, Any]) -> Path:
        """Download the paper PDF."""
        import requests

        pdf_url = metadata["pdf_url"]
        pdf_path = self.output_dir / f"{metadata['arxiv_id'].replace('/', '_')}.pdf"

        self._log(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

        with open(pdf_path, "wb") as f:
            f.write(response.content)

        self._log(f"PDF saved to: {pdf_path}")
        return pdf_path

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text content from PDF.

        2026-04-30: pdfplumber 優先。pdfplumber は table 抽出が強く、
        論文 Table 1 / Table 2 の数値（FID 1.13 等）を markdown 化して
        取り出せる。失敗時は PyPDF2 にフォールバック。
        """
        if _HAS_PDFPLUMBER:
            try:
                return self._extract_with_pdfplumber(pdf_path)
            except Exception as e:
                self._log(f"pdfplumber 抽出失敗、PyPDF2 にフォールバック: {e}")
        return self._extract_with_pypdf2(pdf_path)

    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract using pdfplumber, including table extraction as markdown."""
        text_parts: list[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # 1) Plain text
                page_text = page.extract_text() or ""
                if page_text:
                    text_parts.append(page_text)
                # 2) Tables — markdown 化して text に追加
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                for tbl_idx, tbl in enumerate(tables, 1):
                    if not tbl or not any(any(cell for cell in row if cell) for row in tbl):
                        continue
                    md = self._table_to_markdown(tbl)
                    if md:
                        text_parts.append(
                            f"\n[Table extracted from page {page_num} (table {tbl_idx})]\n{md}\n"
                        )
        return "\n".join(text_parts)

    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Fallback: PyPDF2 のみ（table 抽出は不可）。"""
        reader = PdfReader(str(pdf_path))
        text_parts: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts)

    @staticmethod
    def _table_to_markdown(table: list[list[str | None]]) -> str:
        """Convert a 2D table (list of rows) to a markdown table string."""
        if not table:
            return ""
        # Normalize: replace None with empty, strip
        rows = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in table]
        # Drop fully empty rows
        rows = [r for r in rows if any(c for c in r)]
        if not rows:
            return ""
        # Use widest row as column count
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        # Build markdown
        header = "| " + " | ".join(rows[0]) + " |"
        sep = "| " + " | ".join(["---"] * ncol) + " |"
        body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n".join([header, sep] + body)

    def _analyze_paper(self, paper_text: str, metadata: dict) -> dict[str, Any]:
        """Use Claude to extract key experimental information from the paper.

        Extracts:
            - Main experimental claims (with numerical results)
            - Datasets used
            - Model architecture description
            - Training procedure
            - Reported metrics
        """
        # Truncate paper text to fit context
        # 2026-04-30: 80000 -> 160000 に拡張（Claude Sonnet 4 の context window で十分扱える）
        truncated_text = paper_text[:160000]

        prompt = f"""Analyze this ML paper and extract the following information in a structured format.

Paper title: {metadata['title']}
Paper abstract: {metadata['abstract']}

Full paper text:
{truncated_text}

Please extract and return the following in a structured format:

1. EXPERIMENTAL CLAIMS: List the main quantitative claims (e.g., "achieves 95.2% accuracy on ImageNet"). Include:
   - The metric name
   - The reported value
   - The dataset/benchmark
   - The model variant (if multiple)
   - Which table/figure it appears in (if identifiable)

2. DATASETS: List all datasets used in experiments with:
   - Dataset name
   - How it's used (training, evaluation, both)
   - Any specific splits or configurations mentioned

3. MODEL ARCHITECTURE: Describe the model architecture including:
   - Base architecture (Transformer, CNN, etc.)
   - Key hyperparameters (layers, hidden dim, etc.)
   - Any novel components

4. TRAINING PROCEDURE: Detail the training setup:
   - Optimizer and learning rate
   - Batch size
   - Number of epochs/steps
   - Hardware used (GPUs, TPUs)
   - Training time (if mentioned)

5. CODE REFERENCES: Any URLs to code repositories, GitHub links, or implementation references mentioned in the paper.

Format your response as a valid JSON object with keys: claims, datasets, architecture, training, code_references"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text

        # Try to parse as JSON, fall back to raw text
        import json

        try:
            # Find JSON in the response (might be wrapped in markdown)
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                analysis = {"raw_analysis": response_text}
        except json.JSONDecodeError:
            analysis = {"raw_analysis": response_text}

        return analysis

    def _log(self, message: str):
        """Log if verbose mode is on."""
        if self.verbose:
            print(f"[PaperFetcher] {message}")
