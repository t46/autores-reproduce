# autores-reproduce

Automated ML paper reproduction pipeline. Takes an arXiv URL and attempts to reproduce the paper's experimental results end-to-end.

**📊 Live results**: <https://t46.github.io/autores-showcase/reproduce-deep-dive.html>
(Pipeline-Level Rethink experiment: 5 papers × 4 modes = 20 evaluations)

**🔗 Companion repo**: [t46/autores-showcase](https://github.com/t46/autores-showcase) — UI / 結果ビジュアライゼーション

---

## Quick clone & run

```bash
# 1. Clone both repos
git clone https://github.com/t46/autores-reproduce.git
git clone https://github.com/t46/autores-showcase.git

# 2. Set Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Install
cd autores-reproduce && uv sync

# 4. Run on a paper (default mode)
uv run reproduce https://arxiv.org/abs/2310.03725 --output-dir ./out --no-gpu

# 5. Or run the pipeline-rethink experiment branch
git checkout reproduce/pipeline-rethink-2026-04-30
uv run reproduce https://arxiv.org/abs/2310.03725 \
    --prompt-mode rubric-aware \
    --rubric-path /path/to/paperbench-data/.../rubric.json \
    --paper-cache-dir /path/to/paperbench-data/.../stochastic-interpolants \
    --output-dir ./out --no-gpu
```

---

## Branches

| branch | 内容 |
|---|---|
| `main` | original 5-stage pipeline (1-shot per stage) |
| `reproduce/strengthen-2026-04-30` | 4 つの局所改善 (metric alias 拡張 / success 厳密化 / pdfplumber / max_tokens 緩和) |
| `reproduce/pipeline-rethink-2026-04-30` | strengthen + ARA-findings validation (C-008/C-010/H4) + Rubric-Aware Stage 2 mode + paper-cache-dir |


## How It Works

```
arXiv URL
    |
    v
+-------------------+     +------------------+     +-------------------+
| 1. Paper Fetcher  | --> | 2. Code Finder   | --> | 3. Env Builder    |
| - Download PDF    |     | - GitHub search  |     | - Detect deps     |
| - Extract text    |     | - Clone repo     |     | - Create venv     |
| - Claude: claims  |     | - Or: generate   |     | - Install pkgs    |
+-------------------+     +------------------+     +-------------------+
                                                           |
                                                           v
                          +------------------+     +-------------------+
                          | 5. Verifier      | <-- | 4. Executor       |
                          | - Compare claims |     | - Find entry pt   |
                          | - Score results  |     | - Run w/ timeout  |
                          | - JSON report    |     | - Capture output  |
                          +------------------+     +-------------------+
                                  |
                                  v
                          Reproduction Report
                          (JSON + Markdown)
```

## Installation

```bash
# Clone and install
cd ~/dev/autores-reproduce
uv sync

# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Basic usage
uv run reproduce https://arxiv.org/abs/2301.12345

# With options
uv run reproduce https://arxiv.org/abs/2301.12345 \
    --output-dir ./results \
    --timeout 7200 \
    --no-gpu \
    --verbose
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` / `-o` | `./reproduction_output` | Where to store artifacts |
| `--timeout` / `-t` | `3600` | Max execution time in seconds |
| `--gpu` / `--no-gpu` | `--gpu` | Whether to use GPU if available |
| `--verbose` / `-v` | off | Detailed stage-by-stage logging |
| `--prompt-mode` | `default` | `default` / `ara-fixes` (H4 enforcement) / `rubric-aware` (rubric leaf checklist 注入) — pipeline-rethink branch のみ |
| `--rubric-path` | none | PaperBench `rubric.json` のパス。`--prompt-mode=rubric-aware` のとき必須 |
| `--paper-cache-dir` | none | paperbench-data 形式の dir (paper.pdf + paper.md)。指定すると arXiv API を bypass (429 回避) |

### Pipeline-Level Rethink experiment (pipeline-rethink branch)

```bash
# baseline (default prompt, repo lookup あり)
uv run reproduce https://arxiv.org/abs/2310.03725 --no-gpu

# ara-fixes mode (H4 enforcement, repo lookup skip — generation 直行)
uv run reproduce https://arxiv.org/abs/2310.03725 \
    --prompt-mode ara-fixes \
    --paper-cache-dir /path/to/paperbench-data/.../stochastic-interpolants \
    --no-gpu

# rubric-aware mode (rubric leaf 要件を Stage 2 prompt に直接注入)
uv run reproduce https://arxiv.org/abs/2310.03725 \
    --prompt-mode rubric-aware \
    --rubric-path /path/to/paperbench-data/.../stochastic-interpolants/rubric.json \
    --paper-cache-dir /path/to/paperbench-data/.../stochastic-interpolants \
    --no-gpu
```

**※ Caveat**: `rubric-aware` は judge が見る要件を agent に直接見せる構造で、benchmark 評価としては test-disclosure に近い。詳細は [showcase §6.5](https://t46.github.io/autores-showcase/reproduce-deep-dive.html#rethink-residue) 参照。

### Batch evaluation (PaperBench 5 papers × 4 modes)

```bash
# 5 論文 × {baseline, improved, ara-fixes, rubric-aware} = 20 evaluations
uv run python scripts/batch_paperbench.py \
    --variants baseline improved ara-fixes rubric-aware \
    --paper-fetch-interval 5

# 結果は results/paperbench-batch/<paper>/<variant>/{summary.json, evaluation.json}
```

**前提**: `paperbench-data` を別途 clone しておく必要あり (PaperBench 公式 repo: <https://github.com/openai/preparedness/tree/main/project/paperbench>)。 `batch_paperbench.py` 内 `PAPERBENCH_DATA` 定数を自分の path に書き換える。

## Output

The pipeline produces:
- `report.json` - Machine-readable reproduction report
- `report.md` - Human-readable markdown summary
- `paper/` - Downloaded PDF and extracted data
- `code/` - Cloned or generated source code
- `env/` - Isolated virtual environment
- `execution/` - stdout, stderr, output files

## Pipeline Stages

### 1. Paper Fetcher
Downloads the paper from arXiv and uses Claude to extract:
- Main experimental claims (tables/figures with numbers)
- Datasets used
- Model architecture description
- Training procedure details
- Reported metrics

### 2. Code Finder
Searches for official code in this order:
1. Code URLs referenced in the paper
2. GitHub search by paper title / arXiv ID
3. Falls back to Claude-generated implementation

### 3. Environment Builder
- Detects dependencies from requirements.txt / setup.py / pyproject.toml
- Falls back to scanning imports
- Creates isolated venv with uv (or stdlib venv)
- Handles install failures gracefully

### 4. Executor
- Identifies entry point (main.py, train.py, etc.)
- Detects GPU availability
- Runs with configurable timeout
- Captures all output and extracts metrics

### 5. Verifier
- Compares execution metrics against paper claims
- Configurable tolerance (default: 5% relative)
- Falls back to Claude-based comparison for unstructured results
- Produces overall reproduction score

## Current Limitations

- No Docker-based isolation (venv only)
- No HuggingFace model/dataset hub integration yet
- Generated code is simplified; may not capture all paper details
- Large-scale experiments may need manual batch size tuning
- No multi-GPU support in generated code
- Dataset downloads not automated for proprietary datasets
- Paper PDF text extraction can miss tables/figures

## Requirements

- Python 3.13+
- `uv` package manager
- `ANTHROPIC_API_KEY` environment variable
- `git` (for cloning repositories)
- Optional: CUDA-capable GPU for ML experiments

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Run a specific test
uv run pytest tests/test_pipeline.py::TestVerifier -v
```

## License

MIT
