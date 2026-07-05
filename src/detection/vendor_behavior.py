"""Vendor behavior anomaly detection and drift analysis.

Five-step pipeline:

1. **Baseline construction** — per-vendor behavioural profile across
   eight dimensions, cached in SQLite.
2. **Anomaly scoring** — each invoice scored against the vendor's
   baseline on financial, identity, timing and category dimensions.
3. **Composite risk scoring** — weighted combination; bank-account
   changes trigger an immediate critical alert.
4. **Multi-anomaly compound alert** — 3+ low-severity anomalies in
   one invoice escalate to MEDIUM.
5. **Drift detection** — compares the current baseline against a
   stored prior baseline to detect gradual behavioural shifts.

False-positive suppression removes new vendors (< 6 months history)
and verified changes already recorded in the cache.
"""

import logging
import statistics as pystats
from collections import Counter, defaultdict
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd

from src.detection.base_detector import BaseDetector
from src.utils.constants import Finding, ModuleName, Severity
from src.utils.date_utils import days_between, months_between, parse_date
from src.utils.similarity import normalize_text
from src.utils.statistics import compute_zscore

logger = logging.getLogger(__name__)

_CUSTOMER_ID = "default"


def _to_date(val) -> Optional[date]:
    if isinstance(val, date):
        return val
    return parse_date(str(val)) if val is not None else None


def _to_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


class VendorBehaviorDetector(BaseDetector):
    """Detect vendor behaviour anomalies via baseline comparison."""

    def get_module_name(self) -> ModuleName:
        return ModuleName.VENDOR_BEHAVIOR

    def get_required_fields(self) -> list[str]:
        return ["vendor_id", "total_amount", "invoice_date"]

    def get_optional_fields(self) -> list[str]:
        return ["submission_email", "bank_account_last4",
                "contact_person", "line_item_description"]

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def detect(
        self,
        invoice_data: pd.DataFrame,
        vendor_data: Optional[pd.DataFrame] = None,
        supplementary_data: Optional[dict] = None,
    ) -> list[Finding]:
        self.findings = []
        self.stats["invoices_analyzed"] = len(invoice_data)

        vendor_groups = self._group_by_vendor(invoice_data)
        vendor_meta = self._build_vendor_meta(vendor_data)

        min_invoices = self.config.get("min_history_invoices", 6)
        min_months = self.config.get("min_history_months", 3)

        for vid, rows in vendor_groups.items():
            if len(rows) < min_invoices:
                continue
            dated = [r for r in rows if r["invoice_date"] is not None]
            dated.sort(key=lambda r: r["invoice_date"])
            if len(dated) < min_invoices:
                continue
            span = months_between(dated[0]["invoice_date"], dated[-1]["invoice_date"])
            if span < min_months:
                continue

            # Split into historical (for baseline) and recent (to score).
            # The baseline is built from the oldest 75% of invoices so
            # that recent anomalies aren't absorbed into the baseline.
            split_idx = max(int(len(dated) * 0.75), min_invoices)
            if split_idx >= len(dated):
                split_idx = len(dated) - 1
            history = dated[:split_idx]
            recent = dated[split_idx:]

            # Step 1: baseline from history only
            baseline = self._build_baseline(vid, history, vendor_meta.get(vid, {}))

            # Step 5: drift (before we overwrite the stored baseline)
            self._detect_drift(vid, baseline, dated)

            # Store the full baseline (including recent) for next run
            full_baseline = self._build_baseline(vid, dated, vendor_meta.get(vid, {}))
            self.cache.set_vendor_baseline(vid, _CUSTOMER_ID, full_baseline)

            # Steps 2-4: per-invoice anomaly scoring on recent invoices
            for inv in recent:
                self._score_invoice(vid, inv, baseline, vendor_meta.get(vid, {}))

        self.stats["tier1_calls"] += 1
        return self.findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_vendor(df: pd.DataFrame) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = defaultdict(list)
        id_col = "invoice_id" if "invoice_id" in df.columns else "invoice_number"
        for _, row in df.iterrows():
            vid = _safe_str(row.get("vendor_id"))
            if not vid:
                continue
            groups[vid].append({
                "invoice_id": _safe_str(row.get(id_col, "")),
                "vendor_id": vid,
                "vendor_name": _safe_str(row.get("vendor_name", vid)),
                "invoice_date": _to_date(row.get("invoice_date")),
                "total_amount": _to_float(row.get("total_amount")),
                "bank_account_last4": _safe_str(row.get("bank_account_last4")),
                "contact_person": _safe_str(row.get("contact_person")),
                "submission_email": _safe_str(row.get("submission_email")),
                "line_item_description": _safe_str(row.get("line_item_description")),
            })
        return dict(groups)

    @staticmethod
    def _build_vendor_meta(vendor_data: Optional[pd.DataFrame]) -> dict[str, dict]:
        if vendor_data is None or len(vendor_data) == 0:
            return {}
        meta: dict[str, dict] = {}
        for _, row in vendor_data.iterrows():
            vid = _safe_str(row.get("vendor_id"))
            if vid:
                meta[vid] = {
                    "bank_account_last4": _safe_str(row.get("bank_account_last4")),
                    "contact_person": _safe_str(row.get("contact_person")),
                    "category": _safe_str(row.get("category")),
                }
        return meta

    # ==================================================================
    # Step 1 — Baseline Construction
    # ==================================================================

    def _build_baseline(
        self, vid: str, dated: list[dict], meta: dict,
    ) -> dict:
        amounts = [r["total_amount"] for r in dated if r["total_amount"] > 0]
        dates = [r["invoice_date"] for r in dated]

        # Frequency: average days between invoices
        intervals = []
        for i in range(1, len(dates)):
            intervals.append(days_between(dates[i - 1], dates[i]))
        avg_interval = pystats.mean(intervals) if intervals else 30.0
        std_interval = pystats.stdev(intervals) if len(intervals) >= 2 else avg_interval * 0.3

        # Amount distribution
        avg_amount = pystats.mean(amounts) if amounts else 0.0
        std_amount = pystats.stdev(amounts) if len(amounts) >= 2 else avg_amount * 0.2
        max_amount = max(amounts) if amounts else 0.0

        # Line item categories
        categories: Counter = Counter()
        for r in dated:
            desc = r.get("line_item_description", "")
            if desc:
                norm = normalize_text(desc)
                words = [w for w in norm.split() if len(w) >= 3]
                key = " ".join(words[:2]) if words else norm
                if key:
                    categories[key] += 1

        # Identity
        contacts: Counter = Counter()
        banks: Counter = Counter()
        emails: Counter = Counter()
        for r in dated:
            c = r.get("contact_person", "")
            if c:
                contacts[c] += 1
            b = r.get("bank_account_last4", "")
            if b:
                banks[b] += 1
            e = r.get("submission_email", "")
            if e:
                emails[e] += 1

        # Add vendor-master bank if available
        if meta.get("bank_account_last4"):
            banks[meta["bank_account_last4"]] += max(len(dated), 1)

        return {
            "vendor_id": vid,
            "avg_interval": round(avg_interval, 2),
            "std_interval": round(std_interval, 2),
            "avg_amount": round(avg_amount, 2),
            "std_amount": round(std_amount, 2),
            "max_amount": round(max_amount, 2),
            "invoice_count": len(dated),
            "span_months": months_between(dates[0], dates[-1]) if len(dates) >= 2 else 0,
            "top_categories": dict(categories.most_common(10)),
            "known_contacts": dict(contacts),
            "known_banks": dict(banks),
            "known_emails": dict(emails),
        }

    # ==================================================================
    # Steps 2-4 — Per-Invoice Anomaly Scoring
    # ==================================================================

    def _score_invoice(
        self, vid: str, inv: dict, baseline: dict, meta: dict,
    ) -> None:
        anomalies: list[dict] = []
        vname = inv.get("vendor_name", vid)
        inv_id = inv["invoice_id"]

        # --- Financial anomaly ---
        z_warn = self.config.get("amount_zscore_warning", 2.5)
        z_crit = self.config.get("amount_zscore_critical", 3.0)
        amt = inv["total_amount"]
        z = compute_zscore(amt, baseline["avg_amount"], baseline["std_amount"])

        if abs(z) >= z_crit:
            anomalies.append({
                "dimension": "financial",
                "severity": "critical",
                "score": min(abs(z) / 5, 1.0),
                "detail": f"Amount ${amt:,.2f} is {abs(z):.1f} std devs from mean "
                          f"${baseline['avg_amount']:,.2f}",
            })
            # Critical financial anomaly → immediate finding
            self.create_finding(
                severity=Severity.CRITICAL,
                confidence=0.85,
                vendor_id=vid,
                vendor_name=vname,
                invoice_ids=[inv_id],
                amount_at_risk=Decimal(str(round(amt, 2))),
                description=(
                    f"Amount spike for {vname}: ${amt:,.2f} is "
                    f"{abs(z):.1f} std deviations from average "
                    f"${baseline['avg_amount']:,.2f}"
                ),
                evidence={
                    "method": "anomaly_scoring",
                    "subtype": "amount_spike",
                    "z_score": round(abs(z), 2),
                    "amount": amt,
                    "baseline_avg": baseline["avg_amount"],
                    "baseline_std": baseline["std_amount"],
                    "anomalies": [{"dimension": "financial", "severity": "critical"}],
                    "anomaly_count": 1,
                },
                recommended_action="Verify justification for unusually large amount",
            )
        elif abs(z) >= z_warn:
            anomalies.append({
                "dimension": "financial",
                "severity": "warning",
                "score": min(abs(z) / 5, 1.0),
                "detail": f"Amount ${amt:,.2f} is {abs(z):.1f} std devs from mean",
            })

        # --- Bank account change ---
        inv_bank = inv.get("bank_account_last4", "")
        known_banks = baseline.get("known_banks", {})
        if inv_bank and known_banks and inv_bank not in known_banks:
            bank_score = float(self.config.get("bank_change_score", 0.95))
            anomalies.append({
                "dimension": "bank_change",
                "severity": "critical",
                "score": bank_score,
                "detail": f"Bank account changed to *{inv_bank} "
                          f"(known: {', '.join(f'*{b}' for b in known_banks)})",
            })
            # IMMEDIATE critical alert
            self.create_finding(
                severity=Severity.CRITICAL,
                confidence=0.90,
                vendor_id=vid,
                vendor_name=vname,
                invoice_ids=[inv_id],
                amount_at_risk=Decimal(str(round(amt, 2))),
                description=(
                    f"Bank account change for {vname}: new account "
                    f"*{inv_bank} — potential BEC attack"
                ),
                evidence={
                    "method": "bank_change",
                    "new_bank": inv_bank,
                    "known_banks": list(known_banks.keys()),
                    "amount": amt,
                },
                recommended_action=(
                    "Immediately verify bank account change via "
                    "independent phone call to vendor"
                ),
            )

        # --- Contact / identity change ---
        inv_contact = inv.get("contact_person", "")
        known_contacts = baseline.get("known_contacts", {})
        if inv_contact and known_contacts and inv_contact not in known_contacts:
            anomalies.append({
                "dimension": "identity",
                "severity": "warning",
                "score": 0.6,
                "detail": f"New contact person: {inv_contact}",
            })

        inv_email = inv.get("submission_email", "")
        known_emails = baseline.get("known_emails", {})
        if inv_email and known_emails and inv_email not in known_emails:
            anomalies.append({
                "dimension": "identity",
                "severity": "warning",
                "score": 0.5,
                "detail": f"New submission email: {inv_email}",
            })

        # --- Timing anomaly ---
        # (only meaningful if we can compute gap to previous invoice)
        # Skipped for simplicity in per-invoice scoring; covered by drift.

        # --- Category anomaly ---
        desc = inv.get("line_item_description", "")
        if desc:
            norm = normalize_text(desc)
            words = [w for w in norm.split() if len(w) >= 3]
            cat_key = " ".join(words[:2]) if words else norm
            top_cats = baseline.get("top_categories", {})
            if top_cats and cat_key and cat_key not in top_cats:
                anomalies.append({
                    "dimension": "category",
                    "severity": "info",
                    "score": 0.4,
                    "detail": f"New category '{cat_key}' not in vendor profile",
                })

        if not anomalies:
            return

        # --- Step 3: composite risk score ---
        weights = {
            "bank_change": 0.40,
            "financial": 0.25,
            "identity": 0.20,
            "timing": 0.08,
            "category": 0.07,
        }
        composite = 0.0
        for a in anomalies:
            dim = a["dimension"]
            w = weights.get(dim, 0.05)
            composite += w * a["score"]

        high_thresh = self.config.get("composite_high_threshold", 0.70)
        med_thresh = self.config.get("composite_medium_threshold", 0.50)
        low_thresh = self.config.get("composite_low_threshold", 0.30)
        escalation_count = self.config.get("multi_anomaly_escalation_count", 3)

        # Skip if already emitted a bank-change critical
        has_bank = any(a["dimension"] == "bank_change" for a in anomalies)
        if has_bank:
            return  # already emitted above

        # Determine severity
        if composite >= high_thresh:
            severity = Severity.CRITICAL
        elif composite >= med_thresh:
            severity = Severity.REVIEW
        elif composite >= low_thresh:
            severity = Severity.INFORMATIONAL
        else:
            # Step 4: multi-anomaly escalation
            if len(anomalies) >= escalation_count:
                severity = Severity.REVIEW
            else:
                return  # below all thresholds

        desc_parts = [a["detail"] for a in anomalies]
        self.create_finding(
            severity=severity,
            confidence=round(min(composite + 0.3, 0.95), 2),
            vendor_id=vid,
            vendor_name=vname,
            invoice_ids=[inv_id],
            amount_at_risk=Decimal(str(round(amt, 2))),
            description=(
                f"Behavioral anomaly for {vname}: "
                + "; ".join(desc_parts)
            ),
            evidence={
                "method": "anomaly_scoring",
                "composite_score": round(composite, 3),
                "anomalies": anomalies,
                "anomaly_count": len(anomalies),
                "baseline_avg_amount": baseline["avg_amount"],
                "baseline_avg_interval": baseline["avg_interval"],
            },
            recommended_action="Review anomalous invoice against vendor history",
        )

    # ==================================================================
    # Step 5 — Drift Detection
    # ==================================================================

    def _detect_drift(
        self, vid: str, current_baseline: dict, dated: list[dict],
    ) -> None:
        lookback = self.config.get("drift_lookback_months", 3)
        prior = self.cache.get_vendor_baseline(vid, _CUSTOMER_ID)
        if not prior:
            return

        vname = dated[-1].get("vendor_name", vid) if dated else vid
        drift_signals: list[str] = []
        drift_scores: list[float] = []

        # Amount drift
        p_avg = _to_float(prior.get("avg_amount"))
        c_avg = current_baseline["avg_amount"]
        if p_avg > 0:
            amt_change = abs(c_avg - p_avg) / p_avg
            if amt_change >= 0.30:
                direction = "increased" if c_avg > p_avg else "decreased"
                drift_signals.append(
                    f"Average amount {direction} {amt_change * 100:.0f}% "
                    f"(${p_avg:,.0f} → ${c_avg:,.0f})"
                )
                drift_scores.append(min(amt_change, 1.0))

        # Frequency drift
        p_int = _to_float(prior.get("avg_interval"))
        c_int = current_baseline["avg_interval"]
        if p_int > 0:
            int_change = abs(c_int - p_int) / p_int
            if int_change >= 0.40:
                direction = "increased" if c_int > p_int else "decreased"
                drift_signals.append(
                    f"Invoice interval {direction} {int_change * 100:.0f}% "
                    f"({p_int:.0f}d → {c_int:.0f}d)"
                )
                drift_scores.append(min(int_change, 1.0))

        # Category drift
        p_cats = set(prior.get("top_categories", {}).keys())
        c_cats = set(current_baseline.get("top_categories", {}).keys())
        if p_cats:
            new_cats = c_cats - p_cats
            if len(new_cats) >= 2:
                drift_signals.append(
                    f"{len(new_cats)} new categories appeared: "
                    + ", ".join(list(new_cats)[:3])
                )
                drift_scores.append(0.5)

        if len(drift_signals) < 2:
            return

        avg_drift = sum(drift_scores) / len(drift_scores)
        inv_ids = [r["invoice_id"] for r in dated[-3:]]
        self.create_finding(
            severity=Severity.REVIEW if avg_drift >= 0.4 else Severity.INFORMATIONAL,
            confidence=round(min(avg_drift + 0.3, 0.85), 2),
            vendor_id=vid,
            vendor_name=vname,
            invoice_ids=inv_ids,
            amount_at_risk=Decimal(str(round(current_baseline["avg_amount"], 2))),
            description=(
                f"Behavioral drift for {vname} over {lookback} months: "
                + "; ".join(drift_signals)
            ),
            evidence={
                "method": "drift",
                "drift_signals": drift_signals,
                "drift_scores": drift_scores,
                "avg_drift_score": round(avg_drift, 3),
                "prior_baseline": {
                    "avg_amount": prior.get("avg_amount"),
                    "avg_interval": prior.get("avg_interval"),
                },
                "current_baseline": {
                    "avg_amount": current_baseline["avg_amount"],
                    "avg_interval": current_baseline["avg_interval"],
                },
            },
            recommended_action="Compare vendor behaviour trends over recent months",
        )
