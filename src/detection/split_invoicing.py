"""Split invoicing detection — identify approval-threshold circumvention.

Five detection methods:

1. **Threshold clustering** — statistical over-representation of invoice
   amounts just below an approval threshold.
2. **Temporal aggregation** — sliding-window analysis finding clusters of
   small invoices whose aggregate exceeds the next threshold tier.
3. **PO / project-linked splits** — invoices referencing the same PO or
   project that individually stay below a threshold but sum above it.
4. **Benford's Law** — per-vendor first-digit anomaly detection.
5. **Pattern change** — historical average-amount drop with constant or
   increasing total spend.
"""

import logging
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd

from src.detection.base_detector import BaseDetector
from src.utils.constants import Finding, ModuleName, Severity
from src.utils.date_utils import days_between, get_quarter, parse_date
from src.utils.statistics import benfords_test, detect_clustering_below_threshold

logger = logging.getLogger(__name__)


def _to_date(val) -> Optional[date]:
    if isinstance(val, date):
        return val
    return parse_date(str(val)) if val is not None else None


def _to_decimal(val) -> Decimal:
    try:
        return Decimal(str(val))
    except Exception:
        return Decimal("0")


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


class SplitInvoiceDetector(BaseDetector):
    """Detect invoice splitting to circumvent approval thresholds."""

    def get_module_name(self) -> ModuleName:
        return ModuleName.SPLIT_INVOICING

    def get_required_fields(self) -> list[str]:
        return ["vendor_id", "total_amount", "invoice_date"]

    def get_optional_fields(self) -> list[str]:
        return ["po_number", "project_code", "line_item_description"]

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

        # Aggregate to invoice level
        inv_df = self._aggregate_invoices(invoice_data)

        thresholds = self._get_thresholds(supplementary_data or {})

        # Run all five methods
        self._detect_threshold_clustering(inv_df, thresholds)
        self._detect_temporal_aggregation(inv_df, thresholds)
        self._detect_po_project_linked_splits(inv_df, thresholds)
        self._detect_benfords_anomaly(inv_df)
        self._detect_pattern_change(inv_df)

        self.stats["tier1_calls"] += 5
        return self.findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_invoices(df: pd.DataFrame) -> pd.DataFrame:
        """Collapse to one row per invoice (sum line totals)."""
        id_col = "invoice_id" if "invoice_id" in df.columns else "invoice_number"
        keep_cols = [
            c for c in [
                id_col, "invoice_number", "vendor_id", "vendor_name",
                "invoice_date", "total_amount", "po_number", "project_code",
                "line_item_description",
            ] if c in df.columns
        ]
        return df[keep_cols].drop_duplicates(subset=[id_col])

    def _get_thresholds(self, supplementary_data: dict) -> list[Decimal]:
        """Return sorted approval thresholds as Decimals."""
        raw = supplementary_data.get("approval_thresholds")
        if raw:
            vals = []
            for t in raw:
                amt = t.get("max_amount") if isinstance(t, dict) else getattr(t, "max_amount", None)
                if amt is not None:
                    vals.append(Decimal(str(amt)))
            if vals:
                return sorted(vals)

        defaults = self.config.get("default_approval_thresholds", [5000, 25000, 100000])
        return sorted(Decimal(str(d)) for d in defaults)

    def _vendor_groups(self, df: pd.DataFrame) -> dict[str, list[dict]]:
        """Group invoice rows by vendor_id."""
        groups: dict[str, list[dict]] = defaultdict(list)
        id_col = "invoice_id" if "invoice_id" in df.columns else "invoice_number"
        for _, row in df.iterrows():
            vid = _safe_str(row.get("vendor_id"))
            if not vid:
                continue
            groups[vid].append({
                "invoice_id": _safe_str(row.get(id_col)),
                "vendor_id": vid,
                "vendor_name": _safe_str(row.get("vendor_name", vid)),
                "invoice_date": _to_date(row.get("invoice_date")),
                "total_amount": _to_decimal(row.get("total_amount")),
                "po_number": _safe_str(row.get("po_number")),
                "project_code": _safe_str(row.get("project_code")),
                "line_item_description": _safe_str(row.get("line_item_description")),
            })
        return groups

    # ==================================================================
    # Method 1: Threshold Clustering
    # ==================================================================

    def _detect_threshold_clustering(
        self, df: pd.DataFrame, thresholds: list[Decimal],
    ) -> None:
        band_pct = self.config.get("below_threshold_band_pct", 10) / 100

        vendor_groups = self._vendor_groups(df)
        for vid, rows in vendor_groups.items():
            if len(rows) < 20:
                continue

            amounts = [r["total_amount"] for r in rows]
            vname = rows[0]["vendor_name"]
            inv_ids = [r["invoice_id"] for r in rows]

            for threshold in thresholds:
                result = detect_clustering_below_threshold(
                    amounts, threshold, band_pct,
                )
                if result["significant"] and result["count_in_band"] >= 4 and result.get("p_value", 1.0) < 0.01:
                    lower = threshold * Decimal(str(1 - band_pct))
                    self.create_finding(
                        severity=Severity.REVIEW,
                        confidence=0.80,
                        vendor_id=vid,
                        vendor_name=vname,
                        invoice_ids=inv_ids,
                        amount_at_risk=Decimal(str(
                            result["count_in_band"]
                        )) * threshold,
                        description=(
                            f"{result['count_in_band']} invoices from {vname} "
                            f"cluster in ${lower:,.0f}–${threshold:,.0f} band "
                            f"(expected {result['expected_count']:.1f}, "
                            f"p={result['p_value']:.4f})"
                        ),
                        evidence={
                            "method": "threshold_clustering",
                            "threshold": float(threshold),
                            "count_in_band": result["count_in_band"],
                            "expected_count": result["expected_count"],
                            "chi_squared": result["chi_squared"],
                            "p_value": result["p_value"],
                        },
                        recommended_action=(
                            "Review whether invoices are being structured "
                            "to stay below the approval threshold"
                        ),
                    )

    # ==================================================================
    # Method 2: Temporal Aggregation (Sliding Window)
    # ==================================================================

    def _detect_temporal_aggregation(
        self, df: pd.DataFrame, thresholds: list[Decimal],
    ) -> None:
        windows = self.config.get("temporal_windows_days", [7, 14, 30])
        min_cluster = self.config.get("min_invoices_in_cluster", 3)

        vendor_groups = self._vendor_groups(df)
        seen_clusters: set[frozenset[str]] = set()

        for vid, rows in vendor_groups.items():
            dated = [r for r in rows if r["invoice_date"] is not None]
            if len(dated) < min_cluster:
                continue
            dated.sort(key=lambda r: r["invoice_date"])
            vname = dated[0]["vendor_name"]

            # Track invoice IDs already flagged by a smaller window
            flagged_inv_ids: set[str] = set()

            for window_days in windows:
                for i in range(len(dated)):
                    window_start = dated[i]["invoice_date"]
                    window_end = window_start + timedelta(days=window_days)

                    cluster = [
                        r for r in dated
                        if window_start <= r["invoice_date"] <= window_end
                    ]
                    if len(cluster) < min_cluster:
                        continue

                    cluster_ids = frozenset(r["invoice_id"] for r in cluster)
                    if cluster_ids in seen_clusters:
                        continue

                    # Skip if all invoices in this cluster were already
                    # flagged by a smaller window
                    if cluster_ids.issubset(flagged_inv_ids):
                        continue

                    cluster_amounts = [r["total_amount"] for r in cluster]
                    aggregate = sum(cluster_amounts)

                    for t_idx, threshold in enumerate(thresholds):
                        all_below = all(a < threshold for a in cluster_amounts)
                        # Aggregate exceeds this threshold (or the next tier)
                        next_thresh = (
                            thresholds[t_idx + 1]
                            if t_idx + 1 < len(thresholds)
                            else threshold * 2
                        )
                        if all_below and aggregate >= threshold:
                            seen_clusters.add(cluster_ids)
                            flagged_inv_ids.update(cluster_ids)
                            avg_pct_below = float(
                                sum(
                                    (threshold - a) / threshold * 100
                                    for a in cluster_amounts
                                ) / len(cluster_amounts)
                            )
                            confidence = 0.75
                            if avg_pct_below < 5:
                                confidence = 0.85
                            self.create_finding(
                                severity=Severity.REVIEW,
                                confidence=confidence,
                                vendor_id=vid,
                                vendor_name=vname,
                                invoice_ids=list(cluster_ids),
                                amount_at_risk=aggregate,
                                description=(
                                    f"{len(cluster)} invoices from {vname} "
                                    f"within {window_days} days total "
                                    f"${aggregate:,.2f} — each below "
                                    f"${threshold:,.0f} threshold"
                                ),
                                evidence={
                                    "method": "temporal_aggregation",
                                    "window_days": window_days,
                                    "cluster_count": len(cluster),
                                    "aggregate_amount": float(aggregate),
                                    "threshold": float(threshold),
                                    "amounts": [float(a) for a in cluster_amounts],
                                    "avg_pct_below_threshold": round(avg_pct_below, 2),
                                },
                                recommended_action=(
                                    "Verify whether this work should have been "
                                    "submitted as a single purchase"
                                ),
                            )
                            break  # only flag once per cluster

    # ==================================================================
    # Method 3: PO / Project-Linked Splits
    # ==================================================================

    def _detect_po_project_linked_splits(
        self, df: pd.DataFrame, thresholds: list[Decimal],
    ) -> None:
        vendor_groups = self._vendor_groups(df)

        for vid, rows in vendor_groups.items():
            vname = rows[0]["vendor_name"]

            # Group by PO or project code
            link_groups: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                po = r["po_number"]
                proj = r["project_code"]
                key = po or proj
                if key:
                    link_groups[key].append(r)

            # Construction keywords that indicate legitimate progress billing
            _CONSTRUCTION_SUPPRESS = {
                "progress", "draw", "retainage", "milestone", "phase",
                "application for payment", "pay app", "schedule of values",
                "change order", "co #", "requisition",
            }

            for link_key, group in link_groups.items():
                if len(group) < 2:
                    continue

                amounts = [r["total_amount"] for r in group]
                total = sum(amounts)
                inv_ids = [r["invoice_id"] for r in group]

                # Suppress if descriptions contain construction progress keywords
                all_descs = " ".join(
                    r.get("line_item_description", "").lower() for r in group
                )
                if any(kw in all_descs for kw in _CONSTRUCTION_SUPPRESS):
                    continue

                for threshold in thresholds:
                    all_below = all(a < threshold for a in amounts)
                    if all_below and total >= threshold:
                        self.create_finding(
                            severity=Severity.REVIEW,  # Downgraded from CRITICAL
                            confidence=0.75,  # Reduced from 0.85
                            vendor_id=vid,
                            vendor_name=vname,
                            invoice_ids=inv_ids,
                            amount_at_risk=total,
                            description=(
                                f"{len(group)} invoices from {vname} "
                                f"linked by {link_key} total "
                                f"${total:,.2f} — each individually "
                                f"below ${threshold:,.0f}"
                            ),
                            evidence={
                                "method": "po_project_linked",
                                "link_key": link_key,
                                "invoice_count": len(group),
                                "aggregate_amount": float(total),
                                "threshold": float(threshold),
                                "amounts": [float(a) for a in amounts],
                            },
                            recommended_action=(
                                "Review whether these invoices represent "
                                "a single engagement split to avoid approval"
                            ),
                        )
                        break  # one threshold per group

    # ==================================================================
    # Method 4: Benford's Law Anomaly
    # ==================================================================

    def _detect_benfords_anomaly(self, df: pd.DataFrame) -> None:
        min_sample = self.config.get("benfords_min_sample", 50)
        mad_threshold = self.config.get("benfords_mad_first_digit", 0.015)

        vendor_groups = self._vendor_groups(df)
        for vid, rows in vendor_groups.items():
            if len(rows) < min_sample:
                continue
            amounts = [r["total_amount"] for r in rows]
            vname = rows[0]["vendor_name"]

            result = benfords_test(amounts, digits=1)
            if result["significant"] and result["mad"] >= mad_threshold:
                self.create_finding(
                    severity=Severity.INFORMATIONAL,
                    confidence=0.60,
                    vendor_id=vid,
                    vendor_name=vname,
                    invoice_ids=[r["invoice_id"] for r in rows[:10]],
                    amount_at_risk=Decimal("0"),
                    description=(
                        f"Invoice amounts from {vname} deviate from "
                        f"Benford's Law (MAD={result['mad']:.4f}, "
                        f"p={result['p_value']:.4f}) — may indicate "
                        f"amount manipulation"
                    ),
                    evidence={
                        "method": "benfords_anomaly",
                        "chi_squared": result["chi_squared"],
                        "p_value": result["p_value"],
                        "mad": result["mad"],
                        "observed": {
                            str(k): round(v, 4)
                            for k, v in result["observed_distribution"].items()
                        },
                        "expected": {
                            str(k): round(v, 4)
                            for k, v in result["expected_distribution"].items()
                        },
                        "sample_size": len(amounts),
                    },
                    recommended_action=(
                        "Investigate invoice amount distribution for "
                        "signs of intentional structuring"
                    ),
                )

    # ==================================================================
    # Method 5: Pattern Change
    # ==================================================================

    def _detect_pattern_change(self, df: pd.DataFrame) -> None:
        drop_pct = self.config.get("historical_amount_drop_pct", 30)

        vendor_groups = self._vendor_groups(df)
        for vid, rows in vendor_groups.items():
            dated = [r for r in rows if r["invoice_date"] is not None]
            if len(dated) < 8:
                continue

            # Assign quarters
            quarter_data: dict[str, list[dict]] = defaultdict(list)
            for r in dated:
                q = get_quarter(r["invoice_date"])
                quarter_data[q].append(r)

            quarters = sorted(quarter_data.keys())
            if len(quarters) < 3:
                continue

            current_q = quarters[-1]
            prior_qs = quarters[:-1]

            current_invs = quarter_data[current_q]
            prior_invs = [inv for q in prior_qs for inv in quarter_data[q]]

            if not current_invs or not prior_invs:
                continue

            current_avg = float(sum(r["total_amount"] for r in current_invs)) / len(current_invs)
            prior_avg = float(sum(r["total_amount"] for r in prior_invs)) / len(prior_invs)
            current_total = float(sum(r["total_amount"] for r in current_invs))
            prior_avg_total = float(sum(r["total_amount"] for r in prior_invs)) / len(prior_qs)

            if prior_avg == 0:
                continue

            avg_change_pct = (current_avg - prior_avg) / prior_avg * 100
            vname = dated[0]["vendor_name"]

            # Average dropped but total spend stayed same or increased
            if avg_change_pct <= -drop_pct and current_total >= prior_avg_total * 0.8:
                self.create_finding(
                    severity=Severity.REVIEW,
                    confidence=0.70,
                    vendor_id=vid,
                    vendor_name=vname,
                    invoice_ids=[r["invoice_id"] for r in current_invs],
                    amount_at_risk=Decimal(str(round(current_total, 2))),
                    description=(
                        f"Average invoice from {vname} dropped "
                        f"{abs(avg_change_pct):.0f}% (${prior_avg:,.0f} → "
                        f"${current_avg:,.0f}) but total spend maintained — "
                        f"possible split into smaller invoices"
                    ),
                    evidence={
                        "method": "pattern_change",
                        "subtype": "average_drop",
                        "current_avg": round(current_avg, 2),
                        "prior_avg": round(prior_avg, 2),
                        "change_pct": round(avg_change_pct, 2),
                        "current_total": round(current_total, 2),
                        "prior_avg_total": round(prior_avg_total, 2),
                        "current_count": len(current_invs),
                    },
                    recommended_action=(
                        "Compare current invoicing pattern against "
                        "historical norms for this vendor"
                    ),
                )

            # Frequency increase without spend increase
            current_freq = len(current_invs)
            prior_avg_freq = len(prior_invs) / len(prior_qs)
            if prior_avg_freq > 0 and current_freq > prior_avg_freq * 1.5:
                freq_increase = (current_freq - prior_avg_freq) / prior_avg_freq * 100
                if current_total <= prior_avg_total * 1.2:
                    self.create_finding(
                        severity=Severity.INFORMATIONAL,
                        confidence=0.60,
                        vendor_id=vid,
                        vendor_name=vname,
                        invoice_ids=[r["invoice_id"] for r in current_invs],
                        amount_at_risk=Decimal("0"),
                        description=(
                            f"Invoice frequency from {vname} increased "
                            f"{freq_increase:.0f}% ({prior_avg_freq:.0f}/qtr → "
                            f"{current_freq}/qtr) without proportional "
                            f"spend increase"
                        ),
                        evidence={
                            "method": "pattern_change",
                            "subtype": "frequency_increase",
                            "current_freq": current_freq,
                            "prior_avg_freq": round(prior_avg_freq, 1),
                            "freq_increase_pct": round(freq_increase, 1),
                            "current_total": round(current_total, 2),
                            "prior_avg_total": round(prior_avg_total, 2),
                        },
                        recommended_action=(
                            "Investigate why invoicing frequency increased"
                        ),
                    )
