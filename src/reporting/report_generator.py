"""Generate actionable reports from adjudicated findings.

Produces four report types:

1. **Executive summary** — JSON + HTML overview for leadership.
2. **Detailed findings** — JSON + CSV with full evidence.
3. **Vendor risk report** — JSON + HTML per-vendor risk profiles.
4. **Recovery opportunities** — JSON + CSV with gain-share calculation.
"""

import csv
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from src.utils.constants import DataQualityReport, Finding, Severity

logger = logging.getLogger(__name__)

_SEVERITY_COLORS = {
    "critical": "#dc3545",
    "review": "#fd7e14",
    "informational": "#0d6efd",
}

_GAIN_SHARE_PCT = Decimal("0.25")


def _dec(v) -> float:
    return float(v) if v else 0.0


class ReportGenerator:
    """Generate JSON, CSV, and HTML reports from findings."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def generate_all_reports(
        self,
        findings: list[Finding],
        vendor_scores: dict,
        invoice_scores: dict,
        alert_summary: dict,
        data_quality: Optional[DataQualityReport] = None,
    ) -> list[str]:
        """Generate all four report types. Returns list of output paths."""
        paths = []
        paths.extend(self.generate_executive_summary(
            findings, vendor_scores, alert_summary, data_quality))
        paths.extend(self.generate_detailed_findings(findings))
        paths.extend(self.generate_vendor_risk_report(vendor_scores, findings))
        paths.extend(self.generate_recovery_report(findings))
        return paths

    # ==================================================================
    # 1. Executive Summary
    # ==================================================================

    def generate_executive_summary(
        self,
        findings: list[Finding],
        vendor_scores: dict,
        alert_summary: dict,
        data_quality: Optional[DataQualityReport] = None,
    ) -> list[str]:
        active = [f for f in findings if not f.suppressed]

        by_severity: dict[str, int] = defaultdict(int)
        by_module: dict[str, int] = defaultdict(int)
        for f in active:
            by_severity[f.severity.value] += 1
            by_module[f.module.value] += 1

        total_risk = sum(_dec(f.amount_at_risk) for f in active)

        top_vendors = sorted(
            vendor_scores.values(),
            key=lambda v: -v.get("score", 0),
        )[:5]

        recovery = sorted(
            [f for f in active if f.confidence >= 0.70],
            key=lambda f: -_dec(f.amount_at_risk),
        )[:5]

        dq = None
        if data_quality:
            dq = {
                "total_invoices": data_quality.total_invoices,
                "unique_vendors": data_quality.unique_vendors,
                "readiness_score": data_quality.overall_readiness_score,
                "issues": data_quality.data_quality_issues[:5],
            }

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "findings_by_severity": dict(by_severity),
            "total_amount_at_risk": round(total_risk, 2),
            "top_risk_vendors": top_vendors,
            "top_recovery_opportunities": [
                {"vendor": f.vendor_name, "amount": _dec(f.amount_at_risk),
                 "description": f.description[:120]}
                for f in recovery
            ],
            "module_summary": dict(by_module),
            "data_quality": dq,
            "alert_summary": {
                k: (float(v) if isinstance(v, Decimal) else v)
                for k, v in alert_summary.items()
                if k != "top_critical_alerts"
            },
        }

        json_path = os.path.join(self.output_dir, "executive_summary.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

        html_path = os.path.join(self.output_dir, "executive_summary.html")
        self._write_html(html_path, "Executive Summary", self._exec_html(payload))

        return [json_path, html_path]

    # ==================================================================
    # 2. Detailed Findings
    # ==================================================================

    def generate_detailed_findings(self, findings: list[Finding]) -> list[str]:
        active = sorted(
            [f for f in findings if not f.suppressed],
            key=lambda f: (
                -_SEVERITY_ORDER.get(f.severity, 0),
                -_dec(f.amount_at_risk),
            ),
        )

        rows = [self._finding_to_row(f) for f in active]

        json_path = os.path.join(self.output_dir, "detailed_findings.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, default=str)

        csv_path = os.path.join(self.output_dir, "detailed_findings.csv")
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)

        return [json_path, csv_path]

    # ==================================================================
    # 3. Vendor Risk Report
    # ==================================================================

    def generate_vendor_risk_report(
        self, vendor_scores: dict, findings: list[Finding],
    ) -> list[str]:
        vendor_findings: dict[str, list] = defaultdict(list)
        for f in findings:
            if not f.suppressed:
                vendor_findings[f.vendor_id].append(self._finding_to_row(f))

        report = []
        for vid, info in sorted(
            vendor_scores.items(), key=lambda x: -x[1].get("score", 0),
        ):
            tier = info.get("tier")
            report.append({
                "vendor_id": vid,
                "vendor_name": info.get("vendor_name", vid),
                "risk_score": info.get("score", 0),
                "risk_tier": tier.value if hasattr(tier, "value") else str(tier),
                "amount_at_risk": info.get("amount_at_risk", 0),
                "contributing_factors": info.get("contributing_factors", []),
                "findings_count": info.get("findings_count", 0),
                "findings": vendor_findings.get(vid, []),
            })

        json_path = os.path.join(self.output_dir, "vendor_risk_report.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)

        html_path = os.path.join(self.output_dir, "vendor_risk_report.html")
        self._write_html(html_path, "Vendor Risk Report",
                         self._vendor_html(report))

        return [json_path, html_path]

    # ==================================================================
    # 4. Recovery Opportunities
    # ==================================================================

    def generate_recovery_report(self, findings: list[Finding]) -> list[str]:
        candidates = sorted(
            [f for f in findings
             if not f.suppressed and f.confidence >= 0.70
             and _dec(f.amount_at_risk) > 0],
            key=lambda f: -_dec(f.amount_at_risk),
        )

        total = sum(_dec(f.amount_at_risk) for f in candidates)
        gain_share = round(total * float(_GAIN_SHARE_PCT), 2)

        rows = []
        for f in candidates:
            amt = _dec(f.amount_at_risk)
            rows.append({
                "finding_id": f.finding_id,
                "vendor": f.vendor_name,
                "module": f.module.value,
                "severity": f.severity.value,
                "confidence": f.confidence,
                "amount_at_risk": round(amt, 2),
                "gain_share_25pct": round(amt * float(_GAIN_SHARE_PCT), 2),
                "description": f.description[:150],
                "recommended_action": f.recommended_action,
            })

        payload = {
            "total_recoverable": round(total, 2),
            "gain_share_25pct": gain_share,
            "opportunities": rows,
        }

        json_path = os.path.join(self.output_dir, "recovery_opportunities.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

        csv_path = os.path.join(self.output_dir, "recovery_opportunities.csv")
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)

        return [json_path, csv_path]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _finding_to_row(f: Finding) -> dict:
        return {
            "finding_id": f.finding_id,
            "module": f.module.value,
            "severity": f.severity.value,
            "confidence": f.confidence,
            "vendor_id": f.vendor_id,
            "vendor_name": f.vendor_name,
            "invoice_ids": ", ".join(f.invoice_ids[:5]),
            "amount_at_risk": round(_dec(f.amount_at_risk), 2),
            "description": f.description[:200],
            "recommended_action": f.recommended_action,
        }

    @staticmethod
    def _write_html(path: str, title: str, body: str) -> None:
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title} — SilentAuditor</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;margin:2rem;color:#333}}
h1{{color:#1a1a2e}}h2{{border-bottom:2px solid #e0e0e0;padding-bottom:.3rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ddd;padding:.5rem .75rem;text-align:left}}
th{{background:#f5f5f5}}tr:nth-child(even){{background:#fafafa}}
.badge{{padding:.2rem .5rem;border-radius:3px;color:#fff;font-size:.85rem}}
.critical{{background:{_SEVERITY_COLORS['critical']}}}
.review{{background:{_SEVERITY_COLORS['review']}}}
.informational{{background:{_SEVERITY_COLORS['informational']}}}
.footer{{margin-top:2rem;color:#888;font-size:.85rem}}
</style></head><body>
<h1>{title}</h1>
{body}
<div class="footer">Generated by SilentAuditor</div>
</body></html>"""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)

    @staticmethod
    def _exec_html(payload: dict) -> str:
        sev = payload.get("findings_by_severity", {})
        total = payload.get("total_amount_at_risk", 0)
        parts = [
            f"<h2>Overview</h2>",
            f"<p>Total amount at risk: <strong>${total:,.2f}</strong></p>",
            "<table><tr><th>Severity</th><th>Count</th></tr>",
        ]
        for s in ["critical", "review", "informational"]:
            cnt = sev.get(s, 0)
            parts.append(
                f'<tr><td><span class="badge {s}">{s.upper()}</span></td>'
                f'<td>{cnt}</td></tr>'
            )
        parts.append("</table>")

        parts.append("<h2>Top Risk Vendors</h2><table>"
                     "<tr><th>Vendor</th><th>Score</th><th>Findings</th></tr>")
        for v in payload.get("top_risk_vendors", []):
            parts.append(
                f'<tr><td>{v.get("vendor_name","")}</td>'
                f'<td>{v.get("score",0)}</td>'
                f'<td>{v.get("findings_count",0)}</td></tr>'
            )
        parts.append("</table>")
        return "\n".join(parts)

    @staticmethod
    def _vendor_html(report: list[dict]) -> str:
        parts = [
            "<table><tr><th>Vendor</th><th>Score</th><th>Tier</th>"
            "<th>Amount at Risk</th><th>Findings</th></tr>",
        ]
        for v in report:
            tier = v.get("risk_tier", "low")
            cls = tier if tier in _SEVERITY_COLORS else ""
            parts.append(
                f'<tr><td>{v["vendor_name"]}</td>'
                f'<td>{v["risk_score"]}</td>'
                f'<td><span class="badge {cls}">{tier.upper()}</span></td>'
                f'<td>${v.get("amount_at_risk",0):,.2f}</td>'
                f'<td>{v["findings_count"]}</td></tr>'
            )
        parts.append("</table>")
        return "\n".join(parts)


_SEVERITY_ORDER = {Severity.CRITICAL: 3, Severity.REVIEW: 2, Severity.INFORMATIONAL: 1}
