#!/usr/bin/env python3
"""Improve generated code based on PaperBench evaluation failures.

Reads the evaluation results, identifies failed requirements, and uses Claude
to generate improved code that addresses the specific failures.

Usage:
    uv run python scripts/improve_code.py \
        --submission ./prev-iter/code \
        --evaluation ./prev-iter/evaluation.json \
        --rubric ./rubric.json \
        --paper-dir ./paper-data \
        --output-dir ./new-iter/code \
        --model claude-sonnet-4-20250514
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import anthropic


def load_evaluation(eval_path: str) -> dict:
    with open(eval_path) as f:
        return json.load(f)


def load_paper_content(paper_dir: str) -> str:
    """Load paper markdown content via LFS smudge if needed."""
    paper_path = Path(paper_dir) / "paper.md"

    # Check if it's an LFS pointer
    content = paper_path.read_text()
    if content.startswith("version https://git-lfs.github.com/spec/v1"):
        # Try to smudge it
        try:
            result = subprocess.run(
                ["git", "lfs", "smudge"],
                input=content.encode(),
                capture_output=True,
                cwd=paper_dir,
                timeout=30,
            )
            if result.returncode == 0:
                content = result.stdout.decode()
        except Exception:
            pass

    # Truncate to reasonable size for context
    if len(content) > 30000:
        content = content[:30000] + "\n... [truncated]"

    return content


def load_addendum(paper_dir: str) -> str:
    """Load author's addendum."""
    addendum_path = Path(paper_dir) / "addendum.md"
    content = addendum_path.read_text()

    if content.startswith("version https://git-lfs.github.com/spec/v1"):
        try:
            result = subprocess.run(
                ["git", "lfs", "smudge"],
                input=content.encode(),
                capture_output=True,
                cwd=paper_dir,
                timeout=30,
            )
            if result.returncode == 0:
                content = result.stdout.decode()
        except Exception:
            content = "(addendum not available - LFS fetch failed)"

    return content


def load_submission_code(submission_dir: str) -> dict[str, str]:
    """Load all code files as a dict of filename -> content."""
    code = {}
    for ext in ["*.py", "*.txt", "*.yaml", "*.yml", "*.toml"]:
        for file in Path(submission_dir).glob(ext):
            code[file.name] = file.read_text()
    return code


def get_failed_requirements(evaluation: dict, threshold: int = 50) -> list[dict]:
    """Get requirements that scored below threshold."""
    failed = []
    for node in evaluation.get("scored_nodes", []):
        if node["score"] < threshold:
            failed.append(node)
    return failed


def prioritize_failures(failed: list[dict]) -> list[dict]:
    """Sort failures by likely impact and feasibility of fixing."""
    # Group by fine-grained category for coherent improvements
    priority_order = [
        "Dataset and Model Acquisition",  # Easy wins - just use correct dataset
        "Method Implementation",           # Core improvements
        "Experimental Setup",              # Training details
        "Evaluation",                      # Metrics/evaluation
    ]

    def sort_key(item):
        cat = item.get("finegrained_task_category", "")
        try:
            return priority_order.index(cat)
        except ValueError:
            return len(priority_order)

    return sorted(failed, key=sort_key)


def generate_improved_code(
    client: anthropic.Anthropic,
    current_code: dict[str, str],
    failed_requirements: list[dict],
    paper_content: str,
    addendum: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict[str, str]:
    """Use Claude to improve the code based on failures."""

    # Format current code
    code_str = "\n".join(f"=== {name} ===\n{content}" for name, content in current_code.items())

    # Format failed requirements (top 20 most important)
    failures_str = "\n".join(
        f"- [{node['score']}/100] {node['requirements']}\n  Reason: {node['reasoning']}"
        for node in failed_requirements[:20]
    )

    prompt = f"""You are improving a code implementation for reproducing an ML paper. The current code was evaluated against a rubric and several requirements failed.

## Author's Addendum (hints for reproduction)
{addendum}

## Failed Requirements (scored below 50/100)
{failures_str}

## Current Code
{code_str}

## Paper Content (relevant sections)
{paper_content[:15000]}

## Instructions

Rewrite ALL the code files to fix the failed requirements. Key fixes needed:
1. Use ImageNet dataset (via HuggingFace datasets library), not CIFAR-10
2. Implement the EXACT masking strategy from the paper (64 tiles, probability 0.3, same mask for all channels)
3. Implement the EXACT coupling formula: x_0 = mask * x_1 + (1-mask) * noise
4. Add class conditioning channel (uniform fill with class value)
5. Use step-based training (200,000 steps) not epoch-based
6. Implement both Dependent Coupling AND Uncoupled Interpolant models
7. Use the correct interpolant direction: alpha_t = t, beta_t = 1-t for inpainting
8. Mask the output velocity to ensure unmasked pixels stay fixed
9. Implement super-resolution with proper downsampling/upsampling
10. Use forward Euler for sampling as in Algorithm 2

Output the complete improved code for EACH file. Format your response as:

```filename.py
<complete file content>
```

for each file. Include ALL files needed (requirements.txt, model.py, train.py, etc.)."""

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse response to extract files
    text = response.content[0].text
    improved_code = {}

    # Extract code blocks with filenames
    import re
    # Match ```filename.ext or ```python filename.ext patterns
    blocks = re.findall(r'```(\S+\.(?:py|txt|yaml|yml|toml|sh|cfg))\n(.*?)```', text, re.DOTALL)

    if not blocks:
        # Try alternative format: ```language\n# filename.ext\n...
        blocks = re.findall(r'```(?:python|text|yaml|toml|bash)?\n(.*?)```', text, re.DOTALL)
        # This fallback is less reliable, so prefer the first format

    for filename, content in blocks:
        improved_code[filename] = content.strip()

    # If no files extracted, try harder
    if not improved_code:
        # Split by === filename === markers
        parts = re.split(r'===\s*(\S+\.(?:py|txt|yaml|yml|toml))\s*===', text)
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                filename = parts[i]
                content = parts[i + 1].strip()
                # Remove code fence markers if present
                content = re.sub(r'^```\w*\n', '', content)
                content = re.sub(r'\n```$', '', content)
                improved_code[filename] = content

    return improved_code


def main():
    parser = argparse.ArgumentParser(description="Improve code based on evaluation failures")
    parser.add_argument("--submission", required=True, help="Path to current submission code")
    parser.add_argument("--evaluation", required=True, help="Path to evaluation.json")
    parser.add_argument("--rubric", required=True, help="Path to rubric.json")
    parser.add_argument("--paper-dir", required=True, help="Path to paper data directory")
    parser.add_argument("--output-dir", required=True, help="Path to write improved code")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model for improvement")
    args = parser.parse_args()

    # Load everything
    print("Loading evaluation results...")
    evaluation = load_evaluation(args.evaluation)

    print("Loading current code...")
    current_code = load_submission_code(args.submission)

    print("Loading paper content...")
    paper_content = load_paper_content(args.paper_dir)
    addendum = load_addendum(args.paper_dir)

    # Identify failures
    failed = get_failed_requirements(evaluation, threshold=50)
    failed = prioritize_failures(failed)

    print(f"Found {len(failed)} failed requirements (score < 50)")
    print(f"Top failures by category:")
    cats = {}
    for f in failed:
        cat = f.get("finegrained_task_category", "Unknown")
        cats[cat] = cats.get(cat, 0) + 1
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Generate improvements
    print("\nGenerating improved code with Claude...")
    client = anthropic.Anthropic()

    improved_code = generate_improved_code(
        client=client,
        current_code=current_code,
        failed_requirements=failed,
        paper_content=paper_content,
        addendum=addendum,
        model=args.model,
    )

    # Write improved code
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if improved_code:
        print(f"\nWriting {len(improved_code)} improved files:")
        for filename, content in improved_code.items():
            filepath = output_path / filename
            filepath.write_text(content)
            print(f"  {filename} ({len(content)} chars)")
    else:
        print("WARNING: No improved code generated. Copying original.")
        shutil.copytree(args.submission, args.output_dir, dirs_exist_ok=True)

    # Also copy any files that weren't in the improvement
    for filename, content in current_code.items():
        if filename not in improved_code:
            filepath = output_path / filename
            if not filepath.exists():
                filepath.write_text(content)
                print(f"  {filename} (unchanged, copied)")

    print("\nDone. Improved code written to:", args.output_dir)


if __name__ == "__main__":
    main()
