# Example Run

## Basic Usage

```bash
# Reproduce a paper from its arXiv URL
uv run reproduce https://arxiv.org/abs/2301.12345

# With options
uv run reproduce https://arxiv.org/abs/2301.12345 \
    --output-dir ./my-reproduction \
    --timeout 7200 \
    --no-gpu \
    --verbose
```

## What Happens

When you run the command, the pipeline goes through 5 stages:

1. **Paper Fetching** - Downloads the PDF, extracts text, uses Claude to identify experimental claims
2. **Code Finding** - Searches for official code on GitHub, falls back to Claude-generated code
3. **Environment Building** - Creates an isolated venv with detected dependencies
4. **Experiment Execution** - Runs the code with appropriate configuration
5. **Result Verification** - Compares execution output against paper claims

## Example Output

```
Starting reproduction pipeline for: https://arxiv.org/abs/2301.12345
Output directory: /path/to/reproduction_output
Timeout: 3600s | GPU: True | Verbose: False
------------------------------------------------------------

JSON report written to: reproduction_output/report.json
Markdown report written to: reproduction_output/report.md

============================================================
REPRODUCTION SUMMARY
============================================================
Paper: Attention Is All You Need
Overall score: 0.75
Status: partial
```

## Output Files

After execution, you'll find in the output directory:

```
reproduction_output/
├── report.json          # Machine-readable results
├── report.md            # Human-readable report
├── paper/
│   └── 2301.12345.pdf   # Downloaded paper
├── code/
│   └── repo/            # Cloned or generated code
├── env/
│   └── venv/            # Virtual environment
└── execution/
    ├── stdout.txt       # Full stdout
    └── stderr.txt       # Full stderr
```

## Report JSON Structure

```json
{
  "arxiv_url": "https://arxiv.org/abs/2301.12345",
  "paper_title": "Example Paper Title",
  "timestamp": "2024-01-15T10:30:00Z",
  "status": "partial",
  "reproduction_score": 0.75,
  "total_duration": 1234.56,
  "stages": [
    {"name": "Paper Fetching", "success": true, "duration": 12.3},
    {"name": "Code Finding", "success": true, "duration": 45.6},
    {"name": "Environment Building", "success": true, "duration": 89.0},
    {"name": "Experiment Execution", "success": true, "duration": 987.6},
    {"name": "Result Verification", "success": true, "duration": 5.2}
  ],
  "claims": [
    {
      "description": "Achieves 95.2% accuracy on CIFAR-10",
      "status": "verified",
      "expected": 95.2,
      "actual": 94.8,
      "difference_pct": 0.42
    },
    {
      "description": "BLEU score of 28.4 on WMT14 EN-DE",
      "status": "untested",
      "reason": "Dataset not available in execution"
    }
  ]
}
```

## Tips

- Set `ANTHROPIC_API_KEY` environment variable before running
- Use `--no-gpu` if you don't have a CUDA-capable GPU
- Increase `--timeout` for papers with long training procedures
- Use `--verbose` to see detailed stage-by-stage progress
- Papers with official code repositories have much higher success rates
