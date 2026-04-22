"""CLI entry point for autores-reproduce.

Usage:
    uv run reproduce <arXiv-URL>
    uv run reproduce https://arxiv.org/abs/2301.12345 --output-dir ./results
"""

import json
import sys
from pathlib import Path

import click

from .pipeline import ReproductionPipeline


@click.command()
@click.argument("arxiv_url")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./reproduction_output",
    help="Directory to store reproduction artifacts and reports.",
)
@click.option(
    "--timeout",
    "-t",
    type=int,
    default=3600,
    help="Maximum time in seconds for experiment execution (default: 3600).",
)
@click.option(
    "--gpu/--no-gpu",
    default=True,
    help="Whether to attempt GPU usage for experiments.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging output.",
)
def main(arxiv_url: str, output_dir: str, timeout: int, gpu: bool, verbose: bool):
    """Reproduce an ML paper from its arXiv URL.

    Takes an arXiv URL and attempts to reproduce the paper's experimental
    results end-to-end. Outputs a structured JSON report and a human-readable
    markdown summary.

    Example:
        uv run reproduce https://arxiv.org/abs/2301.12345
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Starting reproduction pipeline for: {arxiv_url}")
    click.echo(f"Output directory: {output_path.resolve()}")
    click.echo(f"Timeout: {timeout}s | GPU: {gpu} | Verbose: {verbose}")
    click.echo("-" * 60)

    pipeline = ReproductionPipeline(
        arxiv_url=arxiv_url,
        output_dir=output_path,
        timeout=timeout,
        use_gpu=gpu,
        verbose=verbose,
    )

    try:
        report = pipeline.run()
    except Exception as e:
        click.echo(f"\nPipeline failed with error: {e}", err=True)
        # Still produce a failure report
        report = pipeline.generate_failure_report(str(e))

    # Write JSON report
    json_path = output_path / "report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    click.echo(f"\nJSON report written to: {json_path}")

    # Write markdown report
    md_path = output_path / "report.md"
    with open(md_path, "w") as f:
        f.write(format_markdown_report(report))
    click.echo(f"Markdown report written to: {md_path}")

    # Print summary
    click.echo("\n" + "=" * 60)
    click.echo("REPRODUCTION SUMMARY")
    click.echo("=" * 60)
    click.echo(f"Paper: {report.get('paper_title', 'Unknown')}")
    click.echo(f"Overall score: {report.get('reproduction_score', 'N/A')}")
    click.echo(f"Status: {report.get('status', 'Unknown')}")

    if report.get("status") == "failed":
        sys.exit(1)


def format_markdown_report(report: dict) -> str:
    """Format the reproduction report as readable markdown."""
    lines = [
        f"# Reproduction Report: {report.get('paper_title', 'Unknown')}",
        "",
        f"**arXiv URL**: {report.get('arxiv_url', 'N/A')}",
        f"**Date**: {report.get('timestamp', 'N/A')}",
        f"**Status**: {report.get('status', 'Unknown')}",
        f"**Overall Reproduction Score**: {report.get('reproduction_score', 'N/A')}",
        "",
        "## Pipeline Stages",
        "",
    ]

    for stage in report.get("stages", []):
        status_icon = "+" if stage.get("success") else "x"
        lines.append(f"### [{status_icon}] {stage.get('name', 'Unknown Stage')}")
        lines.append("")
        if stage.get("message"):
            lines.append(f"{stage['message']}")
            lines.append("")
        if stage.get("error"):
            lines.append(f"**Error**: {stage['error']}")
            lines.append("")
        lines.append(f"Duration: {stage.get('duration', 'N/A')}s")
        lines.append("")

    if report.get("claims"):
        lines.append("## Claim Verification")
        lines.append("")
        for claim in report["claims"]:
            status = claim.get("status", "unknown")
            icon = {"verified": "+", "failed": "x", "untested": "?"}.get(status, "?")
            lines.append(f"- [{icon}] {claim.get('description', 'Unknown claim')}")
            if claim.get("expected"):
                lines.append(f"  - Expected: {claim['expected']}")
            if claim.get("actual"):
                lines.append(f"  - Actual: {claim['actual']}")
            lines.append("")

    if report.get("error"):
        lines.append("## Error Details")
        lines.append("")
        lines.append(f"```\n{report['error']}\n```")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
