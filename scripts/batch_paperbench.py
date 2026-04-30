#!/usr/bin/env python3
"""Batch evaluate Reproduce Pipeline against multiple PaperBench papers.

設計 (2026-04-30):
- 5 papers × {baseline, improved} = 10 jobs
- baseline = main branch worktree (/private/tmp/reproduce-baseline)
- improved = current dir (reproduce/strengthen-2026-04-30 branch)
- PaperBench Code-Dev mode のみ (GPU 不要、cost 85% 削減)
- Anthropic 429 対策: judge call の間に sleep 0.5s + max_workers=3
- arXiv 429 対策: paper fetch を sequential + 5 min 間隔
- 失敗時 retry: 3 回 backoff (1s/5s/30s)

Usage:
    uv run python scripts/batch_paperbench.py [--dry-run]
    uv run python scripts/batch_paperbench.py --papers stochastic-interpolants robust-clip
    uv run python scripts/batch_paperbench.py --variants improved  # baseline をスキップ
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPRODUCE_DIR_IMPROVED = Path("/Users/s30825/dev/autores/reproduce")
REPRODUCE_DIR_BASELINE = Path("/private/tmp/reproduce-baseline")
PAPERBENCH_DATA = Path("/Users/s30825/dev/autores/paperbench-data/project/paperbench/data/papers")
RESULTS_ROOT = Path("/Users/s30825/dev/autores/reproduce/results/paperbench-batch")
UV_BIN = "/Users/s30825/.local/bin/uv"

# 5 論文 (小規模、5/1 12:00 までに完走可能)
# arxiv_id は WebSearch で確定 (2026-04-30 21:10)
DEFAULT_PAPERS = [
    {"id": "stochastic-interpolants", "arxiv_id": "2310.03725", "nodes": 94},
    {"id": "semantic-self-consistency", "arxiv_id": "2410.07839", "nodes": 100},
    {"id": "sequential-neural-score-estimation", "arxiv_id": "2210.04872", "nodes": 123},
    {"id": "mechanistic-understanding", "arxiv_id": "2401.01967", "nodes": 128},
    {"id": "robust-clip", "arxiv_id": "2402.12336", "nodes": 146},
]


def log(msg: str, prefix: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {prefix}{msg}", flush=True)


def get_arxiv_id(paper_id: str) -> str | None:
    """Look up the real arxiv_id from paperbench-data config.yaml."""
    config_path = PAPERBENCH_DATA / paper_id / "config.yaml"
    if not config_path.exists():
        return None
    try:
        text = config_path.read_text()
        # config.yaml format: simple "id: ..." style
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("arxiv_id:") or line.startswith("arxiv-id:"):
                return line.split(":", 1)[1].strip().strip("'\"")
        # Fallback: search for arXiv ID pattern in paper.md
        paper_md = PAPERBENCH_DATA / paper_id / "paper.md"
        if paper_md.exists():
            import re
            m = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", paper_md.read_text()[:5000])
            if m:
                return m.group(1)
    except Exception as e:
        log(f"  arxiv_id lookup failed for {paper_id}: {e}")
    return None


def run_pipeline_for_paper(
    paper_id: str,
    arxiv_id: str,
    variant: str,
    out_dir: Path,
    timeout_sec: int = 1200,
) -> dict:
    """Run the reproduce pipeline for one paper, in baseline or improved variant.

    Stage 4 (execution) はスキップしたいので、--no-gpu + 短い timeout で対処。
    実行が失敗しても Stage 1/2/Code Generation は完走するので、その code を
    PaperBench evaluator にかける。
    """
    # Track A/B (2026-04-30): 4 variants
    # - baseline: original main, default prompt
    # - improved: 4 strengthening fixes branch, default prompt
    # - ara-fixes: pipeline-rethink branch (=improved + 3 ARA fixes), prompt_mode=ara-fixes
    # - rubric-aware: same branch, prompt_mode=rubric-aware (rubric leaf checklist injected)
    if variant == "baseline":
        repro_dir = REPRODUCE_DIR_BASELINE
        prompt_mode = "default"
    elif variant == "improved":
        repro_dir = REPRODUCE_DIR_IMPROVED
        prompt_mode = "default"
    elif variant == "ara-fixes":
        repro_dir = REPRODUCE_DIR_IMPROVED
        prompt_mode = "ara-fixes"
    elif variant == "rubric-aware":
        repro_dir = REPRODUCE_DIR_IMPROVED
        prompt_mode = "rubric-aware"
    else:
        return {"success": False, "error": f"unknown variant: {variant}"}

    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_out = out_dir / "pipeline_output"
    pipeline_out.mkdir(exist_ok=True)
    log_path = out_dir / "pipeline.log"

    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    cmd = [
        UV_BIN, "run", "--no-project", "python", "-m", "autores_reproduce.cli",
        arxiv_url,
        "--output-dir", str(pipeline_out),
        "--no-gpu",
        "--timeout", "60",  # execution が長く詰まらないように短め
        "--prompt-mode", prompt_mode,
    ]
    if variant == "rubric-aware":
        rubric_path = PAPERBENCH_DATA / paper_id / "rubric.json"
        if rubric_path.exists():
            cmd.extend(["--rubric-path", str(rubric_path)])
        else:
            log(f"  [{paper_id}/{variant}] WARN: rubric not found at {rubric_path}, falling back to ara-fixes mode")
            # downgrade silently to ara-fixes if rubric is missing
            cmd[cmd.index("--prompt-mode") + 1] = "ara-fixes"

    log(f"  [{paper_id}/{variant}] running pipeline ({arxiv_id})...")
    try:
        with log_path.open("w") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(repro_dir),
                env={**os.environ, "PYTHONPATH": str(repro_dir / "src")},
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
        log(f"  [{paper_id}/{variant}] pipeline exit code: {proc.returncode}")
    except subprocess.TimeoutExpired:
        log(f"  [{paper_id}/{variant}] pipeline TIMEOUT after {timeout_sec}s")
        return {"success": False, "error": "pipeline_timeout"}
    except Exception as e:
        log(f"  [{paper_id}/{variant}] pipeline error: {e}")
        return {"success": False, "error": str(e)}

    return {"success": True, "pipeline_out": str(pipeline_out)}


def find_generated_code_dir(pipeline_out: Path) -> Path | None:
    """Locate the directory containing generated code from pipeline output.

    pipeline は code/ 配下に生成コードを置く。優先順位:
    1. code/generated/ (Claude 生成 fallback)
    2. code/<最初の subdir> (GitHub clone)
    """
    code_root = pipeline_out / "code"
    if not code_root.exists():
        return None
    generated = code_root / "generated"
    if generated.exists() and any(generated.glob("*.py")):
        return generated
    # First subdir with python files
    for sub in sorted(code_root.iterdir()):
        if sub.is_dir() and any(sub.glob("*.py")):
            return sub
    return None


def run_evaluator(
    paper_id: str,
    variant: str,
    submission_dir: Path,
    rubric_path: Path,
    eval_out_path: Path,
    repro_dir: Path,
    max_nodes: int | None = None,
    timeout_sec: int = 3600,
) -> dict:
    """Run PaperBench evaluator on the generated code."""
    cmd = [
        UV_BIN, "run", "--no-project", "python", "scripts/evaluate_paperbench.py",
        "--submission", str(submission_dir),
        "--rubric", str(rubric_path),
        "--mode", "code-dev",
        "--output", str(eval_out_path),
    ]
    if max_nodes:
        cmd.extend(["--max-nodes", str(max_nodes)])

    log(f"  [{paper_id}/{variant}] running evaluator ({rubric_path.name})...")
    log_path = eval_out_path.parent / "evaluate.log"
    try:
        with log_path.open("w") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(repro_dir),
                env={**os.environ, "PYTHONPATH": str(repro_dir / "src")},
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
        log(f"  [{paper_id}/{variant}] evaluator exit code: {proc.returncode}")
        if eval_out_path.exists():
            with eval_out_path.open() as f:
                evaluation = json.load(f)
            return {
                "success": True,
                "hierarchical_score": evaluation.get("hierarchical_score"),
                "simple_average_score": evaluation.get("simple_average_score"),
                "num_nodes": evaluation.get("num_nodes_evaluated") or len(evaluation.get("scored_nodes", [])),
            }
        return {"success": False, "error": "evaluation_file_missing"}
    except subprocess.TimeoutExpired:
        log(f"  [{paper_id}/{variant}] evaluator TIMEOUT")
        return {"success": False, "error": "evaluator_timeout"}
    except Exception as e:
        log(f"  [{paper_id}/{variant}] evaluator error: {e}")
        return {"success": False, "error": str(e)}


def process_paper_variant(paper_id: str, arxiv_id: str, variant: str) -> dict:
    """One full job: pipeline → evaluator for one paper × one variant."""
    out_dir = RESULTS_ROOT / paper_id / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    summary = {
        "paper_id": paper_id,
        "variant": variant,
        "arxiv_id": arxiv_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    repro_dir = REPRODUCE_DIR_BASELINE if variant == "baseline" else REPRODUCE_DIR_IMPROVED
    # ara-fixes / rubric-aware も improved branch (= pipeline-rethink branch in this run) を使う

    # Step 1: pipeline
    pipe_result = run_pipeline_for_paper(paper_id, arxiv_id, variant, out_dir)
    summary["pipeline"] = pipe_result
    if not pipe_result.get("success"):
        summary["status"] = "pipeline_failed"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    # Step 2: locate generated code
    pipeline_out = Path(pipe_result["pipeline_out"])
    code_dir = find_generated_code_dir(pipeline_out)
    if not code_dir:
        log(f"  [{paper_id}/{variant}] generated code not found")
        summary["status"] = "no_generated_code"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    summary["code_dir"] = str(code_dir)

    # Step 3: evaluator
    rubric_path = PAPERBENCH_DATA / paper_id / "rubric.json"
    if not rubric_path.exists():
        log(f"  [{paper_id}/{variant}] rubric not found at {rubric_path}")
        summary["status"] = "no_rubric"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    eval_path = out_dir / "evaluation.json"
    eval_result = run_evaluator(
        paper_id, variant, code_dir, rubric_path, eval_path, repro_dir,
    )
    summary["evaluation"] = eval_result
    summary["status"] = "completed" if eval_result.get("success") else "evaluator_failed"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"  [{paper_id}/{variant}] DONE: {summary.get('status')} score={eval_result.get('hierarchical_score')}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="計画だけ表示")
    parser.add_argument("--papers", nargs="*", default=None, help="paper_id を絞る")
    parser.add_argument("--variants", nargs="*", default=["baseline", "improved"])
    parser.add_argument("--paper-fetch-interval", type=int, default=300, help="paper fetch 間隔 (秒)")
    args = parser.parse_args()

    # 1) Resolve arxiv_ids from paperbench-data config (override DEFAULT_PAPERS)
    papers_to_run = []
    candidates = DEFAULT_PAPERS
    if args.papers:
        candidates = [p for p in DEFAULT_PAPERS if p["id"] in args.papers]
        if not candidates:
            candidates = [{"id": pid, "arxiv_id": None, "nodes": None} for pid in args.papers]

    for p in candidates:
        real_id = get_arxiv_id(p["id"])
        arxiv_id = real_id or p.get("arxiv_id")
        if not arxiv_id:
            log(f"  WARN: arxiv_id 不明 for {p['id']}, スキップ")
            continue
        papers_to_run.append({"id": p["id"], "arxiv_id": arxiv_id})

    log(f"対象論文 {len(papers_to_run)} 本:")
    for p in papers_to_run:
        log(f"  - {p['id']} (arxiv: {p['arxiv_id']})")

    if args.dry_run:
        log("DRY RUN — 終了")
        return

    # 2) Sequential paper fetch + parallel evaluation
    # 注: pipeline.run() 自体に paper fetch が含まれるので、ここでは
    # 「同時に複数の paper を fetch しないよう」 sequential に paper を回す。
    # 各 paper 内では variant (baseline / improved) を順次実行。
    all_results = []
    for i, paper in enumerate(papers_to_run):
        log(f"=== [{i+1}/{len(papers_to_run)}] {paper['id']} ===")
        for variant in args.variants:
            try:
                result = process_paper_variant(paper["id"], paper["arxiv_id"], variant)
                all_results.append(result)
            except Exception as e:
                log(f"FATAL for {paper['id']}/{variant}: {e}")
                traceback.print_exc()
                all_results.append({
                    "paper_id": paper["id"],
                    "variant": variant,
                    "status": "fatal_error",
                    "error": str(e),
                })
        # arXiv rate limit 対策: 次の paper まで間隔を置く
        if i < len(papers_to_run) - 1:
            log(f"  sleeping {args.paper_fetch_interval}s before next paper...")
            time.sleep(args.paper_fetch_interval)

    # 3) Master summary
    master = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "papers_count": len(papers_to_run),
        "results": all_results,
    }
    master_path = RESULTS_ROOT / "_master.json"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    master_path.write_text(json.dumps(master, indent=2, ensure_ascii=False))
    log(f"Master summary: {master_path}")

    # Print final table
    log("\n=== Final results ===")
    for r in all_results:
        score = (r.get("evaluation") or {}).get("hierarchical_score")
        log(f"  {r['paper_id']:35s} {r['variant']:10s} status={r.get('status'):20s} score={score}")


if __name__ == "__main__":
    main()
