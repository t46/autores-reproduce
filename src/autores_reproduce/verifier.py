"""Stage 5: Verify reproduction results against paper claims.

Compares experimental results from execution with the claims extracted
from the paper. Generates a reproduction report with verification status
for each claim and an overall reproduction score.
"""

import re
from typing import Any

from anthropic import Anthropic


class Verifier:
    """Verifies reproduction results against paper claims."""

    def __init__(self, tolerance: float = 0.05, verbose: bool = False):
        """Initialize the verifier.

        Args:
            tolerance: Relative tolerance for numerical comparison (default: 5%).
            verbose: Whether to log detailed information.
        """
        self.tolerance = tolerance
        self.verbose = verbose
        self.client = Anthropic()

    def verify(
        self, paper_info: dict[str, Any], execution_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Verify execution results against paper claims.

        Args:
            paper_info: Dict from PaperFetcher with extracted claims.
            execution_info: Dict from Executor with execution results.

        Returns:
            Dict with verification results, claim statuses, and overall score.
        """
        claims = paper_info.get("claims", [])
        metrics = execution_info.get("metrics", {})
        stdout = execution_info.get("stdout", "")

        if not claims:
            self._log("No claims extracted from paper")
            # Try to use Claude to compare results with paper
            if stdout and paper_info.get("paper_text"):
                return self._claude_verification(paper_info, execution_info)
            return {
                "success": True,
                "message": "No specific claims to verify",
                "score": None,
                "claims": [],
            }

        # Verify each claim
        verified_claims = []
        for claim in claims:
            result = self._verify_claim(claim, metrics, stdout)
            verified_claims.append(result)

        # Calculate overall score
        score = self._calculate_score(verified_claims)

        return {
            "success": True,
            "message": f"Verified {len(verified_claims)} claims. Score: {score:.1%}",
            "score": score,
            "claims": verified_claims,
        }

    def _verify_claim(
        self, claim: dict[str, Any], metrics: dict[str, Any], stdout: str
    ) -> dict[str, Any]:
        """Verify a single claim against execution results.

        A claim is verified if:
        1. We can find a corresponding metric in our results
        2. The value is within tolerance of the paper's reported value
        """
        # Extract claim details
        metric_name = claim.get("metric", "").lower()
        expected_value = claim.get("value")
        description = claim.get("description", str(claim))
        dataset = claim.get("dataset", "")

        if expected_value is None:
            return {
                "description": description,
                "status": "untested",
                "reason": "No numerical value in claim",
                "expected": None,
                "actual": None,
            }

        # Try to parse expected value
        try:
            expected = float(expected_value)
        except (ValueError, TypeError):
            # Try to extract number from string
            match = re.search(r"([\d.]+)", str(expected_value))
            if match:
                expected = float(match.group(1))
            else:
                return {
                    "description": description,
                    "status": "untested",
                    "reason": f"Could not parse expected value: {expected_value}",
                    "expected": expected_value,
                    "actual": None,
                }

        # Find corresponding metric in our results
        actual = self._find_matching_metric(metric_name, metrics, stdout)

        if actual is None:
            return {
                "description": description,
                "status": "untested",
                "reason": f"Metric '{metric_name}' not found in execution output",
                "expected": expected,
                "actual": None,
            }

        # Compare values
        if expected == 0:
            is_close = abs(actual) < 0.01
        else:
            relative_diff = abs(actual - expected) / abs(expected)
            is_close = relative_diff <= self.tolerance

        status = "verified" if is_close else "failed"
        relative_diff_pct = (
            abs(actual - expected) / abs(expected) * 100 if expected != 0 else abs(actual) * 100
        )

        return {
            "description": description,
            "status": status,
            "expected": expected,
            "actual": actual,
            "difference_pct": round(relative_diff_pct, 2),
            "tolerance_pct": self.tolerance * 100,
            "dataset": dataset,
        }

    def _find_matching_metric(
        self, metric_name: str, metrics: dict[str, Any], stdout: str
    ) -> float | None:
        """Find a matching metric value from execution results."""
        # Direct match in metrics dict
        metric_name_lower = metric_name.lower().strip()

        # Common aliases
        aliases = {
            "accuracy": ["accuracy", "acc", "test_accuracy", "eval_accuracy"],
            "loss": ["loss", "test_loss", "eval_loss", "val_loss"],
            "f1": ["f1", "f1_score", "f1-score"],
            "precision": ["precision", "prec"],
            "recall": ["recall", "rec"],
            "bleu": ["bleu", "bleu_score"],
            "rouge": ["rouge", "rouge1", "rouge-1"],
            "perplexity": ["perplexity", "ppl"],
            "mse": ["mse", "mean_squared_error"],
            "mae": ["mae", "mean_absolute_error"],
            "auc": ["auc", "auroc", "auc-roc"],
            "map": ["map", "mAP", "mean_average_precision"],
        }

        # Find which alias group the metric belongs to
        matching_keys = [metric_name_lower]
        for group_name, group_aliases in aliases.items():
            if metric_name_lower in group_aliases or metric_name_lower == group_name:
                matching_keys = group_aliases + [group_name]
                break

        # Search in metrics dict
        for key in matching_keys:
            if key in metrics:
                return float(metrics[key])
            # Try case-insensitive
            for m_key, m_val in metrics.items():
                if m_key.lower() == key.lower():
                    return float(m_val)

        # Search in stdout as fallback
        for key in matching_keys:
            pattern = rf"(?:{re.escape(key)})\s*[:=]\s*([\d.]+)"
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def _calculate_score(self, claims: list[dict[str, Any]]) -> float:
        """Calculate overall reproduction score.

        Score = (verified + 0.5 * untested) / total
        Untested claims get partial credit since they weren't disproven.
        """
        if not claims:
            return 0.0

        verified = sum(1 for c in claims if c["status"] == "verified")
        untested = sum(1 for c in claims if c["status"] == "untested")
        total = len(claims)

        if total == 0:
            return 0.0

        # Verified claims count full, untested count half, failed count zero
        score = (verified + 0.5 * untested) / total
        return round(score, 4)

    def _claude_verification(
        self, paper_info: dict[str, Any], execution_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Use Claude to compare execution results with paper claims.

        Fallback when structured claims aren't available.
        """
        paper_text = paper_info.get("paper_text", "")[:20000]
        stdout = execution_info.get("stdout", "")[-10000:]
        metrics = execution_info.get("metrics", {})

        prompt = f"""Compare the execution results below with the paper's claimed results.

Paper abstract: {paper_info.get('abstract', '')}

Paper results section (truncated):
{paper_text[-5000:]}

Execution output (last portion):
{stdout}

Extracted metrics: {metrics}

Please analyze:
1. Which claims from the paper can be verified from the execution output?
2. Which claims were NOT reproduced or show different results?
3. Which claims couldn't be tested (insufficient output)?

Provide an overall reproduction score from 0.0 to 1.0.

Format as JSON with keys: score (float), summary (string), claims (list of dicts with description, status, expected, actual)"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text

            import json

            # Try to parse JSON from response
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "success": True,
                    "message": result.get("summary", "Claude verification complete"),
                    "score": result.get("score"),
                    "claims": result.get("claims", []),
                }
        except Exception as e:
            self._log(f"Claude verification failed: {e}")

        return {
            "success": True,
            "message": "Could not perform detailed verification",
            "score": None,
            "claims": [],
        }

    def _log(self, message: str):
        """Log if verbose mode is on."""
        if self.verbose:
            print(f"[Verifier] {message}")
