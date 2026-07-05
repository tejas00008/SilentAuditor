"""Compute unified risk scores at vendor and invoice level.

Aggregates findings from all detection modules into a single 0–100
risk score per vendor and per invoice, using module weights, severity
multipliers, and confidence.
"""

import logging
from collections import defaultdict
from typing import Optional

from src.utils.constants import Finding, ModuleName, RiskTier, Severity

logger = logging.getLogger(__name__)


class RiskScorer:
    """Score vendors and invoices on a unified 0–100 risk scale."""

    MODULE_WEIGHTS: dict[ModuleName, float] = {
        ModuleName.DUPLICATE_DETECTION: 0.15,
        ModuleName.PRICE_CREEP:         0.12,
        ModuleName.PHANTOM_SERVICES:    0.18,
        ModuleName.VENDOR_COLLUSION:    0.20,
        ModuleName.CONTRACT_COMPLIANCE: 0.10,
        ModuleName.VENDOR_BEHAVIOR:     0.15,
        ModuleName.MARKET_PRICE:        0.05,
        ModuleName.SPLIT_INVOICING:     0.05,
    }

    SEVERITY_MULTIPLIERS: dict[Severity, float] = {
        Severity.CRITICAL:      1.0,
        Severity.REVIEW:        0.6,
        Severity.INFORMATIONAL: 0.2,
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_vendors(self, findings: list[Finding]) -> dict[str, dict]:
        """Compute a 0–100 risk score per vendor.

        Returns::

            {vendor_id: {
                score, tier, contributing_factors,
                findings_count, amount_at_risk, vendor_name
            }}
        """
        by_vendor: dict[str, list[Finding]] = defaultdict(list)
        for f in findings:
            if not f.suppressed:
                by_vendor[f.vendor_id].append(f)

        result: dict[str, dict] = {}
        for vid, vfindings in by_vendor.items():
            raw = self._raw_score(vfindings)
            score = self._normalize(raw, len(vfindings))
            tier = self._tier(score)

            modules_hit: dict[str, int] = defaultdict(int)
            for f in vfindings:
                modules_hit[f.module.value] += 1

            factors = [
                f"{mod} ({cnt} finding{'s' if cnt > 1 else ''})"
                for mod, cnt in sorted(modules_hit.items(),
                                       key=lambda x: -x[1])
            ]
            amount = sum(float(f.amount_at_risk) for f in vfindings)
            vname = vfindings[0].vendor_name if vfindings else vid

            result[vid] = {
                "score": round(score, 1),
                "tier": tier,
                "contributing_factors": factors,
                "findings_count": len(vfindings),
                "amount_at_risk": round(amount, 2),
                "vendor_name": vname,
            }

        return result

    def score_invoices(self, findings: list[Finding]) -> dict[str, dict]:
        """Compute a 0–100 risk score per invoice.

        Returns::

            {invoice_id: {score, tier, findings: [finding_id, …],
                          vendor_id, vendor_name}}
        """
        by_inv: dict[str, list[Finding]] = defaultdict(list)
        for f in findings:
            if not f.suppressed:
                for inv_id in f.invoice_ids:
                    by_inv[inv_id].append(f)

        result: dict[str, dict] = {}
        for inv_id, inv_findings in by_inv.items():
            raw = self._raw_score(inv_findings)
            score = self._normalize(raw, len(inv_findings))
            tier = self._tier(score)
            vname = inv_findings[0].vendor_name if inv_findings else ""
            vid = inv_findings[0].vendor_id if inv_findings else ""

            result[inv_id] = {
                "score": round(score, 1),
                "tier": tier,
                "findings": [f.finding_id for f in inv_findings],
                "vendor_id": vid,
                "vendor_name": vname,
            }

        return result

    def get_top_risk_vendors(
        self, vendor_scores: dict[str, dict], n: int = 10,
    ) -> list[dict]:
        """Return the *n* highest-risk vendors."""
        items = [
            {"vendor_id": vid, **info}
            for vid, info in vendor_scores.items()
        ]
        items.sort(key=lambda x: -x["score"])
        return items[:n]

    def get_top_risk_invoices(
        self, invoice_scores: dict[str, dict], n: int = 10,
    ) -> list[dict]:
        """Return the *n* highest-risk invoices."""
        items = [
            {"invoice_id": iid, **info}
            for iid, info in invoice_scores.items()
        ]
        items.sort(key=lambda x: -x["score"])
        return items[:n]

    def get_risk_distribution(
        self, vendor_scores: dict[str, dict],
    ) -> dict[str, int]:
        """Count vendors in each risk tier."""
        dist: dict[str, int] = {t.value: 0 for t in RiskTier}
        for info in vendor_scores.values():
            tier = info["tier"]
            key = tier.value if isinstance(tier, RiskTier) else str(tier)
            dist[key] = dist.get(key, 0) + 1
        return dist

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _raw_score(self, findings: list[Finding]) -> float:
        """Sum of weighted-contribution per finding."""
        total = 0.0
        for f in findings:
            w = self.MODULE_WEIGHTS.get(f.module, 0.05)
            m = self.SEVERITY_MULTIPLIERS.get(f.severity, 0.2)
            total += f.confidence * m * w
        return total

    @staticmethod
    def _normalize(raw: float, count: int) -> float:
        """Map raw score to 0–100.

        Uses a logarithmic-ish curve so that a single high-confidence
        critical finding already scores meaningfully, while many small
        findings can accumulate.
        """
        if raw <= 0:
            return 0.0
        # Scale: a single CRITICAL finding from the highest-weight module
        # (collusion, w=0.20, conf=1.0, sev=1.0) gives raw=0.20 → ~65.
        # Two such findings → ~85.  Three → ~95.
        import math
        scaled = 100 * (1 - math.exp(-raw * 8))
        return min(round(scaled, 1), 100.0)

    @staticmethod
    def _tier(score: float) -> RiskTier:
        if score >= 75:
            return RiskTier.CRITICAL
        if score >= 50:
            return RiskTier.HIGH
        if score >= 25:
            return RiskTier.MEDIUM
        return RiskTier.LOW
