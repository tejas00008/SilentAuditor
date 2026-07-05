"""Generate dashboard-ready JSON for frontend consumption."""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from src.utils.constants import DataQualityReport, Finding, RiskTier

logger = logging.getLogger(__name__)


def _dec(v) -> float:
    return float(v) if v else 0.0


class DashboardDataGenerator:
    """Produce a single JSON blob containing all dashboard data."""

    def generate_dashboard_json(
        self,
        findings: list[Finding],
        vendor_scores: dict,
        invoice_scores: dict,
        alert_summary: dict,
        data_quality: Optional[DataQualityReport] = None,
        output_path: Optional[str] = None,
    ) -> dict:
        """Build and optionally write the dashboard payload."""
        active = [f for f in findings if not f.suppressed]

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": self._serialise_summary(alert_summary),
            "risk_distribution": self._risk_distribution(vendor_scores),
            "findings_by_module": self._by_module_count(active),
            "amount_by_module": self._by_module_amount(active),
            "top_vendors": self._top_vendors(vendor_scores, n=10),
            "top_findings": self._top_findings(active, n=20),
            "timeline": self._timeline(active),
            "data_quality": self._dq_summary(data_quality),
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)

        return payload

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise_summary(alert_summary: dict) -> dict:
        out = {}
        for k, v in alert_summary.items():
            if isinstance(v, Decimal):
                out[k] = float(v)
            elif isinstance(v, list):
                out[k] = v
            else:
                out[k] = v
        return out

    @staticmethod
    def _risk_distribution(vendor_scores: dict) -> dict[str, int]:
        dist: dict[str, int] = {t.value: 0 for t in RiskTier}
        for info in vendor_scores.values():
            tier = info.get("tier")
            key = tier.value if isinstance(tier, RiskTier) else str(tier)
            dist[key] = dist.get(key, 0) + 1
        return dist

    @staticmethod
    def _by_module_count(active: list[Finding]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for f in active:
            counts[f.module.value] += 1
        return dict(counts)

    @staticmethod
    def _by_module_amount(active: list[Finding]) -> dict[str, float]:
        amounts: dict[str, float] = defaultdict(float)
        for f in active:
            amounts[f.module.value] += _dec(f.amount_at_risk)
        return {k: round(v, 2) for k, v in amounts.items()}

    @staticmethod
    def _top_vendors(vendor_scores: dict, n: int = 10) -> list[dict]:
        items = []
        for vid, info in vendor_scores.items():
            tier = info.get("tier")
            items.append({
                "vendor_id": vid,
                "vendor_name": info.get("vendor_name", vid),
                "score": info.get("score", 0),
                "tier": tier.value if isinstance(tier, RiskTier) else str(tier),
                "findings_count": info.get("findings_count", 0),
                "amount_at_risk": info.get("amount_at_risk", 0),
            })
        items.sort(key=lambda x: -x["score"])
        return items[:n]

    @staticmethod
    def _top_findings(active: list[Finding], n: int = 20) -> list[dict]:
        ranked = sorted(active, key=lambda f: -_dec(f.amount_at_risk))
        return [
            {
                "finding_id": f.finding_id,
                "module": f.module.value,
                "severity": f.severity.value,
                "confidence": f.confidence,
                "vendor_name": f.vendor_name,
                "amount_at_risk": _dec(f.amount_at_risk),
                "description": f.description[:150],
            }
            for f in ranked[:n]
        ]

    @staticmethod
    def _timeline(active: list[Finding]) -> list[dict]:
        """Aggregate findings by month for a trend chart."""
        buckets: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "amount": 0.0}
        )
        for f in active:
            if f.created_at:
                key = f.created_at.strftime("%Y-%m")
            else:
                key = "unknown"
            buckets[key]["count"] += 1
            buckets[key]["amount"] += _dec(f.amount_at_risk)

        return [
            {"month": k, "count": v["count"], "amount": round(v["amount"], 2)}
            for k, v in sorted(buckets.items())
        ]

    @staticmethod
    def _dq_summary(dq: Optional[DataQualityReport]) -> Optional[dict]:
        if not dq:
            return None
        return {
            "total_invoices": dq.total_invoices,
            "unique_vendors": dq.unique_vendors,
            "readiness_score": dq.overall_readiness_score,
            "issues_count": len(dq.data_quality_issues),
        }
