"""Categorize findings into tiered alerts for action routing.

Three tiers:

* **CRITICAL** — immediate action (high confidence + high amount, bank
  changes, multi-module corroboration, or severity already CRITICAL).
* **REVIEW** — weekly review (moderate confidence/amount).
* **INFORMATIONAL** — monthly summary (everything else non-suppressed).
"""

import logging
from decimal import Decimal
from typing import Optional

from src.utils.constants import Finding, ModuleName, Severity

logger = logging.getLogger(__name__)


class AlertManager:
    """Categorize findings into prioritised alert tiers."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        tiers = self.config.get("alert_tiers", {})
        crit = tiers.get("critical", {})
        rev = tiers.get("review", {})
        self._crit_conf = float(crit.get("min_confidence", 0.85))
        self._crit_amt = Decimal(str(crit.get("min_amount", 5000)))
        self._rev_conf = float(rev.get("min_confidence", 0.60))
        self._rev_amt = Decimal(str(rev.get("min_amount", 1000)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def categorize_alerts(self, findings: list[Finding]) -> dict[str, list[Finding]]:
        """Sort non-suppressed findings into alert tiers."""
        critical: list[Finding] = []
        review: list[Finding] = []
        informational: list[Finding] = []

        for f in findings:
            if f.suppressed:
                continue
            tier = self._classify(f)
            if tier == "critical":
                critical.append(f)
            elif tier == "review":
                review.append(f)
            else:
                informational.append(f)

        # Sort each tier: amount desc, then confidence desc
        key = lambda f: (-float(f.amount_at_risk), -f.confidence)
        critical.sort(key=key)
        review.sort(key=key)
        informational.sort(key=key)

        return {
            "critical": critical,
            "review": review,
            "informational": informational,
        }

    def generate_alert_summary(self, categorized: dict[str, list[Finding]]) -> dict:
        """Produce a concise summary of the categorized alerts."""
        critical = categorized.get("critical", [])
        review = categorized.get("review", [])
        informational = categorized.get("informational", [])
        all_findings = critical + review + informational

        total_risk = sum(f.amount_at_risk for f in all_findings)
        critical_risk = sum(f.amount_at_risk for f in critical)

        modules = sorted({f.module.value for f in all_findings})
        vendors = {f.vendor_id for f in all_findings}

        top_critical = [
            {
                "finding_id": f.finding_id,
                "vendor": f.vendor_name,
                "description": f.description[:120],
                "amount_at_risk": float(f.amount_at_risk),
                "module": f.module.value,
            }
            for f in critical[:5]
        ]

        return {
            "critical_count": len(critical),
            "review_count": len(review),
            "informational_count": len(informational),
            "total_amount_at_risk": total_risk,
            "critical_amount": critical_risk,
            "top_critical_alerts": top_critical,
            "modules_triggered": modules,
            "vendors_flagged": len(vendors),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _classify(self, f: Finding) -> str:
        # Already CRITICAL from conflict resolver
        if f.severity == Severity.CRITICAL:
            return "critical"

        # Bank account change — always critical
        if (f.module == ModuleName.VENDOR_BEHAVIOR
                and f.evidence.get("method") == "bank_change"):
            return "critical"

        # Multi-module corroboration (2+ correlated finding IDs)
        if len(f.module_correlations) >= 2:
            return "critical"

        try:
            amt = f.amount_at_risk if f.amount_at_risk == f.amount_at_risk else Decimal("0")
        except Exception:
            amt = Decimal("0")
        conf = f.confidence

        # High confidence + high amount
        if conf >= self._crit_conf and amt >= self._crit_amt:
            return "critical"

        # Review band
        if conf >= self._rev_conf and amt >= self._rev_amt:
            return "review"
        if conf >= self._crit_conf and amt >= Decimal("500"):
            return "review"

        return "informational"
