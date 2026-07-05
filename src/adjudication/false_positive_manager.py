"""Learn from user feedback to reduce false positives over time.

Tracks verdicts (confirmed fraud / false positive / legitimate) per
finding, computes per-module FP rates, auto-suppresses known FP
patterns, and suggests threshold adjustments.
"""

import hashlib
import logging
from collections import Counter, defaultdict
from typing import Optional

from src.normalization.cache import CacheManager
from src.utils.constants import (
    FeedbackVerdict,
    Finding,
    ModuleName,
    Severity,
)

logger = logging.getLogger(__name__)

# How many FP verdicts on the same pattern before auto-suppression.
_CUSTOMER_FP_THRESHOLD = 3
_GLOBAL_FP_THRESHOLD = 3

# Default confidence thresholds per module (used for adjustment suggestions).
_DEFAULT_THRESHOLDS: dict[str, float] = {
    ModuleName.DUPLICATE_DETECTION.value: 0.70,
    ModuleName.PRICE_CREEP.value:         0.60,
    ModuleName.PHANTOM_SERVICES.value:    0.60,
    ModuleName.VENDOR_COLLUSION.value:    0.60,
    ModuleName.CONTRACT_COMPLIANCE.value: 0.60,
    ModuleName.VENDOR_BEHAVIOR.value:     0.60,
    ModuleName.MARKET_PRICE.value:        0.50,
    ModuleName.SPLIT_INVOICING.value:     0.60,
}


def _pattern_hash(finding: Finding) -> str:
    """Deterministic hash for a finding's pattern (module + vendor + evidence type)."""
    parts = [
        finding.module.value,
        finding.vendor_id,
        str(finding.evidence.get("method", "")),
        str(finding.evidence.get("check", "")),
        str(finding.evidence.get("layer", "")),
        str(finding.evidence.get("subtype", "")),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class FalsePositiveManager:
    """Learn from user feedback to reduce false positives."""

    def __init__(self, cache: CacheManager) -> None:
        self.cache = cache

    # ------------------------------------------------------------------
    # Auto-suppression
    # ------------------------------------------------------------------

    def apply_suppression_rules(
        self, findings: list[Finding], customer_id: str,
    ) -> list[Finding]:
        """Suppress findings that match known FP patterns."""
        for f in findings:
            if f.suppressed:
                continue

            ph = _pattern_hash(f)
            module = f.module.value

            # Customer-level: same pattern marked FP 3+ times by this customer
            cust_count = self.cache.get_false_positive_count(
                module, ph, customer_id=customer_id,
            )
            if cust_count >= _CUSTOMER_FP_THRESHOLD:
                f.suppressed = True
                f.suppression_reason = (
                    f"Previously marked false positive by customer "
                    f"({cust_count} times)"
                )
                continue

            # Global: same pattern marked FP by 3+ different customers
            global_count = self.cache.get_false_positive_count(module, ph)
            if global_count >= _GLOBAL_FP_THRESHOLD:
                f.suppressed = True
                f.suppression_reason = (
                    f"Known false positive pattern across customers "
                    f"({global_count} reports)"
                )

        return findings

    # ------------------------------------------------------------------
    # Feedback recording
    # ------------------------------------------------------------------

    def record_feedback(
        self,
        finding_id: str,
        customer_id: str,
        verdict: FeedbackVerdict,
        notes: str = "",
    ) -> None:
        """Record user feedback on a finding."""
        self.cache.record_feedback(
            finding_id, customer_id, verdict.value, notes or None,
        )

    # ------------------------------------------------------------------
    # FP rate computation
    # ------------------------------------------------------------------

    def get_false_positive_rate(
        self, customer_id: str, module: Optional[ModuleName] = None,
    ) -> dict[str, dict]:
        """Compute FP rate from feedback data.

        Returns ``{module_value: {total_reviewed, false_positives, fp_rate}}``.
        """
        # Fetch all feedback for the customer
        modules = [module] if module else list(ModuleName)
        result: dict[str, dict] = {}

        for mod in modules:
            mod_val = mod.value
            feedback = self.cache.get_feedback_for_pattern(mod_val, "")
            cust_fb = [fb for fb in feedback if fb.get("customer_id") == customer_id]

            total = len(cust_fb)
            fps = sum(1 for fb in cust_fb if fb.get("verdict") == "false_positive")
            rate = fps / total if total > 0 else 0.0

            result[mod_val] = {
                "total_reviewed": total,
                "false_positives": fps,
                "fp_rate": round(rate, 3),
            }

        return result

    # ------------------------------------------------------------------
    # Common FP patterns
    # ------------------------------------------------------------------

    def get_common_fp_patterns(
        self, customer_id: Optional[str] = None,
    ) -> list[dict]:
        """Return the most frequent FP patterns.

        Each entry: ``{module, pattern_count, sample_finding_id}``.
        """
        # Gather all FP feedback
        all_fb: list[dict] = []
        for mod in ModuleName:
            fb = self.cache.get_feedback_for_pattern(mod.value, "")
            if customer_id:
                fb = [f for f in fb if f.get("customer_id") == customer_id]
            fps = [f for f in fb if f.get("verdict") == "false_positive"]
            for entry in fps:
                entry["_module"] = mod.value
            all_fb.extend(fps)

        # Count by module
        module_counts: Counter = Counter()
        module_samples: dict[str, str] = {}
        for fb in all_fb:
            mod = fb["_module"]
            module_counts[mod] += 1
            if mod not in module_samples:
                module_samples[mod] = fb.get("finding_id", "")

        return [
            {
                "module": mod,
                "pattern_count": cnt,
                "sample_finding_id": module_samples.get(mod, ""),
            }
            for mod, cnt in module_counts.most_common()
        ]

    # ------------------------------------------------------------------
    # Threshold adjustment suggestions
    # ------------------------------------------------------------------

    def suggest_threshold_adjustments(
        self, customer_id: str,
    ) -> dict[str, dict]:
        """Suggest confidence-threshold changes for modules with FP rate > 20%."""
        fp_rates = self.get_false_positive_rate(customer_id)
        suggestions: dict[str, dict] = {}

        for mod_val, info in fp_rates.items():
            if info["total_reviewed"] < 5:
                continue  # not enough data
            if info["fp_rate"] <= 0.20:
                continue

            current = _DEFAULT_THRESHOLDS.get(mod_val, 0.60)
            # Tighten proportionally to how bad the FP rate is
            bump = min(info["fp_rate"] * 0.3, 0.20)
            suggested = round(min(current + bump, 0.95), 2)

            suggestions[mod_val] = {
                "current_threshold": current,
                "suggested_threshold": suggested,
                "fp_rate": info["fp_rate"],
                "total_reviewed": info["total_reviewed"],
            }

        return suggestions

    # ------------------------------------------------------------------
    # Monthly FP report
    # ------------------------------------------------------------------

    def generate_fp_report(self, customer_id: str) -> dict:
        """Full FP report: rates, common patterns, recommendations."""
        rates = self.get_false_positive_rate(customer_id)
        patterns = self.get_common_fp_patterns(customer_id)
        adjustments = self.suggest_threshold_adjustments(customer_id)

        total_reviewed = sum(r["total_reviewed"] for r in rates.values())
        total_fps = sum(r["false_positives"] for r in rates.values())
        overall_rate = total_fps / total_reviewed if total_reviewed else 0.0

        return {
            "customer_id": customer_id,
            "overall_fp_rate": round(overall_rate, 3),
            "total_reviewed": total_reviewed,
            "total_false_positives": total_fps,
            "per_module_rates": rates,
            "common_patterns": patterns,
            "threshold_adjustments": adjustments,
        }
