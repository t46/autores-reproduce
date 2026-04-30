#!/usr/bin/env python3
"""Evaluate a reproduction output against PaperBench rubric.

Uses Claude as the judge (matching PaperBench's LLM-judge approach)
to score generated code against rubric leaf nodes.

Usage:
    uv run python scripts/evaluate_paperbench.py \
        --submission ~/dev/autores/results/stochastic-interpolants/code/generated \
        --rubric ~/dev/autores/paperbench-data/project/paperbench/data/papers/stochastic-interpolants/rubric.json \
        --mode code-dev \
        --output ~/dev/autores/results/stochastic-interpolants/evaluation.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic


def load_rubric(rubric_path: str) -> dict:
    """Load the PaperBench rubric."""
    with open(rubric_path) as f:
        return json.load(f)


def get_leaf_nodes(node: dict, category_filter: str | None = None) -> list[dict]:
    """Extract leaf nodes from the rubric tree, optionally filtering by category."""
    leaves = []
    if not node.get("sub_tasks"):
        if category_filter is None or node.get("task_category") == category_filter:
            leaves.append(node)
    else:
        for sub in node["sub_tasks"]:
            leaves.extend(get_leaf_nodes(sub, category_filter))
    return leaves


def get_weighted_tree(node: dict, category_filter: str | None = None) -> list[dict]:
    """Get leaf nodes with their hierarchical weights computed."""
    def _collect(node, weight_path=None):
        if weight_path is None:
            weight_path = []
        current_weight = node.get("weight", 1)
        new_path = weight_path + [current_weight]

        if not node.get("sub_tasks"):
            if category_filter is None or node.get("task_category") == category_filter:
                return [{"node": node, "weight_path": new_path}]
            return []

        results = []
        for sub in node["sub_tasks"]:
            results.extend(_collect(sub, new_path))
        return results

    return _collect(node)


SKIP_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache", "dist", "build", ".tox", ".eggs"}
MAX_FILE_BYTES = 100 * 1024  # 100 KB
MAX_TOTAL_CHARS = 80_000


def load_submission_code(submission_dir: str) -> str:
    """Load all code files from the submission directory recursively.

    ARA-a C-008 fix (2026-04-30): glob -> rglob so submission/<subdir>/*.py is included.
    Skips .git / venv / __pycache__ etc. Truncates at MAX_TOTAL_CHARS to avoid overflowing judge.
    """
    code_parts: list[str] = []
    submission_path = Path(submission_dir)
    seen_paths: set[Path] = set()
    total_chars = 0

    for ext in ["*.py", "*.txt", "*.yaml", "*.yml", "*.toml", "*.cfg", "*.sh"]:
        for file in sorted(submission_path.rglob(ext)):
            if file in seen_paths:
                continue
            seen_paths.add(file)
            # Skip if any parent directory is in SKIP_DIRS
            if any(part in SKIP_DIRS for part in file.relative_to(submission_path).parts):
                continue
            try:
                size = file.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_BYTES:
                continue
            try:
                content = file.read_text(errors="replace")
            except OSError:
                continue
            rel = file.relative_to(submission_path)
            chunk = f"=== {rel} ===\n{content}\n"
            if total_chars + len(chunk) > MAX_TOTAL_CHARS:
                code_parts.append(f"\n[truncated: {MAX_TOTAL_CHARS} char limit reached, remaining files skipped]\n")
                break
            code_parts.append(chunk)
            total_chars += len(chunk)
        else:
            continue
        break

    return "\n".join(code_parts)


def judge_requirement(
    client: anthropic.Anthropic,
    requirement: str,
    code: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Use Claude as judge to score a single rubric requirement against the code."""
    prompt = f"""You are a judge evaluating whether a code submission satisfies a specific requirement from a paper reproduction rubric.

REQUIREMENT:
{requirement}

SUBMITTED CODE:
{code}

Score this requirement on a scale from 0 to 100:
- 100: The requirement is fully and correctly satisfied
- 75: The requirement is mostly satisfied with minor issues
- 50: The requirement is partially satisfied
- 25: There is a relevant attempt but it's mostly incorrect
- 0: The requirement is not addressed at all

Respond with a JSON object containing:
- "score": integer 0-100
- "reasoning": brief explanation (1-2 sentences)

IMPORTANT: Respond ONLY with the JSON object, no other text."""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Try to parse JSON
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        return result
    except Exception as e:
        return {"score": 0, "reasoning": f"Judge error: {e}"}


def compute_hierarchical_score(scored_nodes: list[dict], rubric: dict, category_filter: str | None = None) -> float:
    """Compute the PaperBench-style hierarchical weighted score.

    Each parent node's score = weighted average of children scores.
    Final score = root score (0-100 scale).
    """
    score_map = {item["id"]: item["score"] for item in scored_nodes}

    def _score_node(node):
        if not node.get("sub_tasks"):
            # Leaf node
            if category_filter and node.get("task_category") != category_filter:
                return None  # Skip non-matching categories
            return score_map.get(node["id"], 0)

        # Internal node: weighted average of children
        child_scores = []
        child_weights = []
        for sub in node["sub_tasks"]:
            s = _score_node(sub)
            if s is not None:
                child_scores.append(s)
                child_weights.append(sub.get("weight", 1))

        if not child_scores:
            return None

        total_weight = sum(child_weights)
        weighted_sum = sum(s * w for s, w in zip(child_scores, child_weights))
        return weighted_sum / total_weight if total_weight > 0 else 0

    result = _score_node(rubric)
    return result if result is not None else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate submission against PaperBench rubric")
    parser.add_argument("--submission", required=True, help="Path to submission code directory")
    parser.add_argument("--rubric", required=True, help="Path to rubric.json")
    parser.add_argument("--mode", choices=["full", "code-dev"], default="code-dev",
                       help="Evaluation mode (code-dev skips execution/results requirements)")
    parser.add_argument("--output", required=True, help="Path to write evaluation results")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model to use as judge")
    parser.add_argument("--max-nodes", type=int, default=None, help="Max leaf nodes to evaluate (for testing)")
    parser.add_argument("--paper-id", default=None, help="Paper id for output metadata (auto-detect from rubric path if omitted)")
    args = parser.parse_args()
    # ARA-a C-010 fix: derive paper_id from rubric path so it isn't hardcoded
    if args.paper_id is None:
        rubric_path = Path(args.rubric)
        # paperbench-data/.../papers/<paper-id>/rubric.json
        if rubric_path.parent.name and rubric_path.parent.parent.name == "papers":
            args.paper_id = rubric_path.parent.name
        else:
            args.paper_id = rubric_path.parent.name or "unknown"

    # Load data
    rubric = load_rubric(args.rubric)
    code = load_submission_code(args.submission)

    # Determine which nodes to evaluate
    category_filter = "Code Development" if args.mode == "code-dev" else None
    leaves = get_leaf_nodes(rubric, category_filter)

    if args.max_nodes:
        leaves = leaves[:args.max_nodes]

    print(f"Evaluating {len(leaves)} leaf nodes in '{args.mode}' mode")
    print(f"Submission: {args.submission}")
    print(f"Code length: {len(code)} chars")
    print("-" * 60)

    # Initialize Claude client
    client = anthropic.Anthropic()

    # Score each leaf node
    scored_nodes = []
    total_score = 0

    for i, leaf in enumerate(leaves):
        req = leaf["requirements"]
        print(f"[{i+1}/{len(leaves)}] {req[:80]}...")

        result = judge_requirement(client, req, code, model=args.model)
        score = result.get("score", 0)
        reasoning = result.get("reasoning", "")

        scored_nodes.append({
            "id": leaf["id"],
            "requirements": req,
            "task_category": leaf.get("task_category"),
            "finegrained_task_category": leaf.get("finegrained_task_category"),
            "score": score,
            "reasoning": reasoning,
        })

        total_score += score
        print(f"  Score: {score}/100 - {reasoning}")

        # Rate limiting
        time.sleep(0.5)

    # Compute hierarchical score
    hierarchical_score = compute_hierarchical_score(scored_nodes, rubric, category_filter)
    simple_avg = total_score / len(scored_nodes) if scored_nodes else 0

    # Compile results
    results = {
        "mode": args.mode,
        "paper_id": args.paper_id,
        "submission_path": args.submission,
        "model_judge": args.model,
        "num_nodes_evaluated": len(scored_nodes),
        "num_nodes_total": len(get_leaf_nodes(rubric, category_filter)),
        "hierarchical_score": round(hierarchical_score, 2),
        "simple_average_score": round(simple_avg, 2),
        "scored_nodes": scored_nodes,
        "score_distribution": {
            "0-25": sum(1 for n in scored_nodes if n["score"] <= 25),
            "26-50": sum(1 for n in scored_nodes if 26 <= n["score"] <= 50),
            "51-75": sum(1 for n in scored_nodes if 51 <= n["score"] <= 75),
            "76-100": sum(1 for n in scored_nodes if n["score"] >= 76),
        },
    }

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Nodes evaluated: {len(scored_nodes)}/{len(get_leaf_nodes(rubric, category_filter))}")
    print(f"Hierarchical score: {hierarchical_score:.1f}%")
    print(f"Simple average: {simple_avg:.1f}%")
    print(f"Distribution: {results['score_distribution']}")
    print(f"\nResults written to: {args.output}")

    return results


if __name__ == "__main__":
    main()
