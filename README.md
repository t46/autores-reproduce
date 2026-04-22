# autores-reproduce

Automated ML paper reproduction pipeline. Takes an arXiv URL and attempts to reproduce the paper's experimental results end-to-end.

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
