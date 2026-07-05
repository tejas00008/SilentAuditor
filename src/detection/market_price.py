"""Market price benchmarking and deviation detection.

Five-step pipeline:

1. **Internal benchmark** — per-category price statistics (median, p25,
   p75, std) across all vendors with 3+ vendors per category.
2. **Cross-client benchmark** — (placeholder) accepts anonymised
   external data and merges it into the internal benchmarks.
3. **Deviation scoring** — flags items priced above p75 + 15%.
4. **Vendor premium index** — flags vendors whose aggregate premium
   across all categories exceeds 15%.
5. **Renegotiation opportunity** — estimates savings if each vendor's
   prices were brought to the category median.
"""

import logging
from collections import defaultdict
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd

from src.detection.base_detector import BaseDetector
from src.utils.constants import Finding, ModuleName, Severity
from src.utils.date_utils import parse_date
from src.utils.similarity import normalize_text

logger = logging.getLogger(__name__)


def _to_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _category_key(desc: str) -> str:
    """Derive a rough category key from a description."""
    norm = normalize_text(desc)
    words = [w for w in norm.split() if len(w) >= 3]
    return " ".join(words[:2]) if words else norm


class MarketPriceDetector(BaseDetector):
    """Detect above-market pricing via internal and external benchmarks."""

    def get_module_name(self) -> ModuleName:
        return ModuleName.MARKET_PRICE

    def get_required_fields(self) -> list[str]:
        return ["vendor_id", "line_item_description", "unit_price",
                "quantity", "invoice_date"]

    def get_optional_fields(self) -> list[str]:
        return ["vendor_region"]

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
        supp = supplementary_data or {}

        items = self._extract_items(invoice_data)
        if not items:
            return self.findings

        # Step 1
        benchmarks = self._build_internal_benchmarks(items)
        self.stats["tier1_calls"] += 1

        # Step 2
        external = supp.get("cross_client_benchmarks")
        if external:
            benchmarks = self._merge_external(benchmarks, external)

        # Step 3
        deviations = self._score_deviations(items, benchmarks)

        # Step 4 — Vendor premium: only annotate existing findings, don't
        # create standalone premium findings (they are cost-savings, not fraud)
        self._compute_vendor_premium_index(items, benchmarks)

        # Step 5 — Renegotiation: DISABLED in fraud pipeline.
        # Renegotiation opportunities are cost-savings intelligence, not fraud
        # indicators. They were producing ~1,500 low-value findings at 0%
        # precision. If needed, expose via a separate savings report.
        # self._estimate_renegotiation(items, benchmarks)

        return self.findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_items(df: pd.DataFrame) -> list[dict]:
        id_col = "invoice_id" if "invoice_id" in df.columns else "invoice_number"
        items: list[dict] = []
        for _, row in df.iterrows():
            vid = _safe_str(row.get("vendor_id"))
            desc = _safe_str(row.get("line_item_description"))
            price = _to_float(row.get("unit_price"))
            qty = _to_float(row.get("quantity"))
            if not vid or not desc or price <= 0:
                continue
            items.append({
                "invoice_id": _safe_str(row.get(id_col, "")),
                "vendor_id": vid,
                "vendor_name": _safe_str(row.get("vendor_name", vid)),
                "description": desc,
                "category": _category_key(desc),
                "unit_price": price,
                "quantity": qty if qty > 0 else 1.0,
                "total": round(price * max(qty, 1.0), 2),
            })
        return items

    # ==================================================================
    # Step 1 — Internal Benchmark
    # ==================================================================

    def _build_internal_benchmarks(
        self, items: list[dict],
    ) -> dict[str, dict]:
        min_vendors = self.config.get("min_vendors_per_category", 3)

        # {category: {vendor_id: [prices]}}
        cat_vendor_prices: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for it in items:
            cat_vendor_prices[it["category"]][it["vendor_id"]].append(
                it["unit_price"]
            )

        benchmarks: dict[str, dict] = {}
        for cat, vendor_prices in cat_vendor_prices.items():
            if len(vendor_prices) < min_vendors:
                continue
            # One representative price per vendor (median of their prices)
            vendor_medians = [
                float(np.median(prices)) for prices in vendor_prices.values()
            ]
            all_prices = [p for prices in vendor_prices.values() for p in prices]
            arr = np.array(all_prices)
            benchmarks[cat] = {
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "p90": float(np.percentile(arr, 90)),
                "std": float(np.std(arr)),
                "mean": float(np.mean(arr)),
                "vendor_count": len(vendor_prices),
                "sample_size": len(all_prices),
                "vendor_medians": vendor_medians,
            }

        return benchmarks

    # ==================================================================
    # Step 2 — Cross-Client Benchmark (placeholder)
    # ==================================================================

    @staticmethod
    def _merge_external(
        internal: dict[str, dict],
        external: dict,
    ) -> dict[str, dict]:
        """Merge external benchmarks into internal, weighted by sample size."""
        merged = dict(internal)
        for cat, ext in external.items():
            if cat in merged:
                int_n = merged[cat]["sample_size"]
                ext_n = ext.get("sample_size", 0)
                total = int_n + ext_n
                if total == 0:
                    continue
                w_int = int_n / total
                w_ext = ext_n / total
                merged[cat] = {
                    "median": merged[cat]["median"] * w_int + ext.get("median", 0) * w_ext,
                    "p25": merged[cat]["p25"] * w_int + ext.get("p25", 0) * w_ext,
                    "p75": merged[cat]["p75"] * w_int + ext.get("p75", 0) * w_ext,
                    "p90": merged[cat].get("p90", 0) * w_int + ext.get("p90", 0) * w_ext,
                    "std": merged[cat]["std"],
                    "mean": merged[cat]["mean"] * w_int + ext.get("mean", 0) * w_ext,
                    "vendor_count": merged[cat]["vendor_count"] + ext.get("vendor_count", 0),
                    "sample_size": total,
                    "vendor_medians": merged[cat].get("vendor_medians", []),
                }
            else:
                merged[cat] = ext

        return merged

    # ==================================================================
    # Step 3 — Deviation Scoring
    # ==================================================================

    def _score_deviations(
        self, items: list[dict], benchmarks: dict[str, dict],
    ) -> list[dict]:
        # Changed from p75+15% to p90+10% to reduce false positives
        deviation_pct_thresh = self.config.get("deviation_flag_above_p90_pct",
                                                self.config.get("deviation_flag_above_p75_pct", 10))
        min_abs_deviation = self.config.get("min_absolute_deviation_dollars", 50)
        min_total_overcharge = self.config.get("min_total_overcharge", 500)
        deviations: list[dict] = []

        # Track vendors that already have item-level findings for premium merge
        vendors_with_deviations: set[str] = set()

        for it in items:
            cat = it["category"]
            bench = benchmarks.get(cat)
            if not bench:
                continue
            price = it["unit_price"]
            p90 = bench.get("p90", bench["p75"] * 1.2)
            median = bench["median"]

            # Use p90 as the threshold base instead of p75
            threshold = p90 * (1 + deviation_pct_thresh / 100)
            if price <= threshold:
                continue

            # Min absolute deviation check: per-unit > $50 OR total > $500
            per_unit_excess = price - median
            total_excess = per_unit_excess * it["quantity"]
            if per_unit_excess < min_abs_deviation and total_excess < min_total_overcharge:
                continue

            deviation_from_median = (
                (price - median) / median * 100 if median > 0 else 0
            )

            severity = Severity.REVIEW
            confidence = 0.75

            self.create_finding(
                severity=severity,
                confidence=confidence,
                vendor_id=it["vendor_id"],
                vendor_name=it["vendor_name"],
                invoice_ids=[it["invoice_id"]],
                amount_at_risk=Decimal(str(round(total_excess, 2))),
                description=(
                    f"Above-market price: {it['vendor_name']} charges "
                    f"${price:,.2f} for '{it['description'][:60]}' — "
                    f"{deviation_from_median:+.0f}% vs category median "
                    f"${median:,.2f}"
                ),
                evidence={
                    "method": "deviation_scoring",
                    "category": cat,
                    "unit_price": price,
                    "median": median,
                    "p90": p90,
                    "deviation_from_median_pct": round(deviation_from_median, 2),
                    "per_unit_excess": round(per_unit_excess, 2),
                    "total_excess": round(total_excess, 2),
                    "vendor_count_in_category": bench["vendor_count"],
                },
                recommended_action="Request competitive quote or negotiate rate",
            )
            vendors_with_deviations.add(it["vendor_id"])
            deviations.append({
                "vendor_id": it["vendor_id"],
                "category": cat,
                "deviation_pct": deviation_from_median,
            })

        # Store for premium index to check
        self._vendors_with_deviations = vendors_with_deviations
        return deviations

    # ==================================================================
    # Step 4 — Vendor Premium Index
    # ==================================================================

    def _compute_vendor_premium_index(
        self, items: list[dict], benchmarks: dict[str, dict],
    ) -> None:
        premium_thresh = self.config.get("vendor_premium_index_threshold", 15)

        # {vendor_id: [(deviation_pct, spend)]}
        vendor_devs: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        for it in items:
            bench = benchmarks.get(it["category"])
            if not bench or bench["median"] <= 0:
                continue
            dev = (it["unit_price"] - bench["median"]) / bench["median"] * 100
            vendor_devs[it["vendor_id"]].append(
                (dev, it["total"], it["vendor_name"])
            )

        # Get set of vendors that already have item-level deviation findings
        vendors_with_deviations = getattr(self, "_vendors_with_deviations", set())

        for vid, entries in vendor_devs.items():
            if not entries:
                continue
            total_spend = sum(e[1] for e in entries)
            if total_spend <= 0:
                continue
            # Spend-weighted average premium
            weighted_prem = sum(e[0] * e[1] for e in entries) / total_spend
            vname = entries[0][2]

            if weighted_prem <= premium_thresh:
                continue

            # Always merge premium data into existing deviation findings
            if vid in vendors_with_deviations:
                for f in self.findings:
                    if (f.vendor_id == vid
                            and f.evidence.get("method") == "deviation_scoring"
                            and not f.suppressed):
                        f.evidence["vendor_premium_pct"] = round(weighted_prem, 2)
                        f.evidence["vendor_total_spend"] = total_spend
                continue

            # Standalone vendor premium findings (no item-level deviations)
            # are cost-savings opportunities, NOT fraud indicators.
            # Skip creating a finding — this vendor is simply expensive
            # but no individual item crossed the p90+10% fraud threshold.
            continue

            # Dead code below preserved for future savings-report feature
            inv_ids = list({it["invoice_id"] for it in items if it["vendor_id"] == vid})[:10]
            savings = round(total_spend * weighted_prem / (100 + weighted_prem), 2)

            self.create_finding(
                severity=Severity.REVIEW,
                confidence=0.75,
                vendor_id=vid,
                vendor_name=vname,
                invoice_ids=inv_ids,
                amount_at_risk=Decimal(str(savings)),
                description=(
                    f"Vendor premium: {vname} averages {weighted_prem:+.1f}% "
                    f"above market across all categories "
                    f"(${total_spend:,.0f} total spend)"
                ),
                evidence={
                    "method": "vendor_premium_index",
                    "weighted_premium_pct": round(weighted_prem, 2),
                    "total_spend": total_spend,
                    "estimated_savings": savings,
                    "categories_compared": len(entries),
                },
                recommended_action=(
                    "Schedule vendor rate review or issue RFP for "
                    "competitive pricing"
                ),
            )

    # ==================================================================
    # Step 5 — Renegotiation Opportunity Estimation
    # ==================================================================

    def _estimate_renegotiation(
        self, items: list[dict], benchmarks: dict[str, dict],
    ) -> None:
        # Aggregate per-vendor savings if brought to median
        vendor_savings: dict[str, dict] = defaultdict(
            lambda: {"savings": 0.0, "vname": "", "inv_ids": set(), "categories": set()}
        )

        for it in items:
            bench = benchmarks.get(it["category"])
            if not bench or bench["median"] <= 0:
                continue
            if it["unit_price"] <= bench["median"]:
                continue

            excess = (it["unit_price"] - bench["median"]) * it["quantity"]
            rec = vendor_savings[it["vendor_id"]]
            rec["savings"] += excess
            rec["vname"] = it["vendor_name"]
            rec["inv_ids"].add(it["invoice_id"])
            rec["categories"].add(it["category"])

        for vid, rec in vendor_savings.items():
            if rec["savings"] < 5000:
                continue

            self.create_finding(
                severity=Severity.INFORMATIONAL,
                confidence=0.65,
                vendor_id=vid,
                vendor_name=rec["vname"],
                invoice_ids=list(rec["inv_ids"])[:10],
                amount_at_risk=Decimal(str(round(rec["savings"], 2))),
                description=(
                    f"Renegotiation opportunity with {rec['vname']}: "
                    f"${rec['savings']:,.0f} potential savings across "
                    f"{len(rec['categories'])} categories if moved to "
                    f"category median pricing"
                ),
                evidence={
                    "method": "renegotiation",
                    "estimated_savings": round(rec["savings"], 2),
                    "categories": list(rec["categories"]),
                },
                recommended_action="Include in next contract renegotiation cycle",
            )
