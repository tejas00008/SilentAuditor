"""Price creep detection over time for vendor line items.

Analyses chronological unit-price series for each (vendor, item) pair
and flags:

* **Cumulative drift** exceeding 6-month or 12-month thresholds.
* **Single-jump spikes** between consecutive invoices.
* **Accelerating trends** where the rate of increase itself grows.
* **Contract-rate breaches** where the current price exceeds the
  contracted rate (with allowed escalation).

Volume changes are checked as context — a genuine mix-shift can
explain price movement and is noted in the evidence.
"""

import logging
from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Optional

import pandas as pd

from src.detection.base_detector import BaseDetector
from src.utils.constants import Finding, ModuleName, Severity
from src.utils.date_utils import months_between, parse_date
from src.utils.similarity import normalize_text
from src.utils.statistics import linear_regression_trend

logger = logging.getLogger(__name__)


def _to_date(val) -> Optional[date]:
    if isinstance(val, date):
        return val
    return parse_date(str(val)) if val is not None else None


def _to_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _to_decimal(val) -> Decimal:
    try:
        return Decimal(str(val))
    except Exception:
        return Decimal("0")


class PriceCreepDetector(BaseDetector):
    """Detect gradual or sudden price increases across vendor line items."""

    def get_module_name(self) -> ModuleName:
        return ModuleName.PRICE_CREEP

    def get_required_fields(self) -> list[str]:
        return ["vendor_id", "line_item_description", "unit_price", "invoice_date"]

    def get_optional_fields(self) -> list[str]:
        return ["quantity", "po_number"]

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

        contracts = (supplementary_data or {}).get("contracts", [])

        # Step 1
        time_series = self._build_price_time_series(invoice_data)
        self.stats["tier1_calls"] += 1

        # Step 2 + 3 + 4
        for (vid, norm_item), series in time_series.items():
            analysis = self._analyze_trend(series)

            # Contract check
            contract_info = None
            if contracts:
                current_price = _to_decimal(series[-1]["unit_price"])
                contract_info = self._check_contract_rates(
                    vid, norm_item, current_price, contracts,
                )

            # Volume context
            volume_note = self._check_volume_changes(series)

            # Build finding if warranted
            self._emit_findings(vid, norm_item, series, analysis,
                                contract_info, volume_note)

        return self.findings

    # ------------------------------------------------------------------
    # Step 1: Build time series
    # ------------------------------------------------------------------

    def _build_price_time_series(self, df: pd.DataFrame) -> dict:
        """Build ``{(vendor_id, normalised_item): [records…]}``."""
        min_points = self.config.get("min_data_points", 4)
        min_months = self.config.get("min_months", 3)

        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for _, row in df.iterrows():
            vid = str(row.get("vendor_id", ""))
            desc = str(row.get("line_item_description", ""))
            price = _to_float(row.get("unit_price"))
            inv_date = _to_date(row.get("invoice_date"))

            if not vid or not desc or price <= 0 or inv_date is None:
                continue

            norm_item = normalize_text(desc)
            if not norm_item:
                continue

            qty = _to_float(row.get("quantity", 1))
            inv_id = str(row.get("invoice_id", row.get("invoice_number", "")))

            groups[(vid, norm_item)].append({
                "date": inv_date,
                "unit_price": price,
                "quantity": qty if qty > 0 else 1.0,
                "invoice_id": inv_id,
                "vendor_name": str(row.get("vendor_name", vid)),
            })

        # Filter and sort
        result: dict[tuple[str, str], list[dict]] = {}
        for key, records in groups.items():
            if len(records) < min_points:
                continue
            records.sort(key=lambda r: r["date"])
            span = months_between(records[0]["date"], records[-1]["date"])
            if span < min_months:
                continue
            result[key] = records

        return result

    # ------------------------------------------------------------------
    # Step 2: Statistical trend analysis
    # ------------------------------------------------------------------

    def _analyze_trend(self, series: list[dict]) -> dict:
        dates = [r["date"] for r in series]
        prices = [r["unit_price"] for r in series]

        # (a) Linear regression
        reg = linear_regression_trend(dates, prices)

        # (b) Cumulative change
        first_price = prices[0]
        last_price = prices[-1]
        if first_price != 0:
            cumulative_pct = (last_price - first_price) / first_price * 100
        else:
            cumulative_pct = 0.0

        # (c) Single-jump detection
        max_jump_pct = 0.0
        max_jump_date: Optional[date] = None
        jumps: list[float] = []
        for i in range(1, len(prices)):
            if prices[i - 1] != 0:
                pct = (prices[i] - prices[i - 1]) / prices[i - 1] * 100
            else:
                pct = 0.0
            jumps.append(pct)
            if abs(pct) > abs(max_jump_pct):
                max_jump_pct = pct
                max_jump_date = dates[i]

        # (d) Acceleration
        is_accelerating = False
        if len(jumps) >= 3:
            last_three = jumps[-3:]
            is_accelerating = (
                last_three[0] > 0
                and last_three[1] > last_three[0]
                and last_three[2] > last_three[1]
            )

        span = months_between(dates[0], dates[-1])

        return {
            "slope": reg["slope"],
            "r_squared": reg["r_squared"],
            "p_value": reg["p_value"],
            "cumulative_change_pct": round(cumulative_pct, 2),
            "max_single_jump_pct": round(max_jump_pct, 2),
            "max_jump_date": max_jump_date,
            "is_accelerating": is_accelerating,
            "data_points": len(series),
            "period_months": span,
            "jumps": jumps,
            "first_price": first_price,
            "last_price": last_price,
        }

    # ------------------------------------------------------------------
    # Step 3: Contract rate comparison
    # ------------------------------------------------------------------

    def _check_contract_rates(
        self,
        vendor_id: str,
        item_desc: str,
        current_price: Decimal,
        contracts: list,
    ) -> Optional[dict]:
        tolerance = Decimal(str(self.config.get("contract_tolerance_pct", 3))) / 100

        # Guard against NaN/invalid Decimal
        try:
            if not (current_price > 0):
                return None
        except Exception:
            return None

        def _cattr(obj: object, key: str, default: object = None) -> object:
            """Read attribute from dict or dataclass."""
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        for contract in contracts:
            c_vid = str(_cattr(contract, "vendor_id", ""))
            if c_vid != vendor_id:
                continue

            rates = _cattr(contract, "rates", [])
            escalation = Decimal(str(_cattr(contract, "annual_escalation_pct", 0) or 0)) / 100
            start_str = _cattr(contract, "contract_start_date", "")
            start = _to_date(start_str) or date(2024, 1, 1)
            years_elapsed = max(months_between(start, date.today()) / 12, 0)

            for rate_entry in rates:
                rate_desc = normalize_text(str(_cattr(rate_entry, "item_description", "")))
                base_rate = Decimal(str(_cattr(rate_entry, "rate", 0)))
                if base_rate <= 0:
                    continue

                # Allow match if the normalised item contains the rate
                # description or vice-versa (partial match is fine for
                # items like "steel rebar" vs "steel rebar #4 grade 60")
                if not (rate_desc in item_desc or item_desc in rate_desc):
                    # Also try word-overlap
                    overlap = set(rate_desc.split()) & set(item_desc.split())
                    if len(overlap) < 2:
                        continue

                # Compute allowed rate with escalation
                allowed = base_rate * (1 + escalation) ** Decimal(str(int(years_elapsed)))
                allowed_with_tol = allowed * (1 + tolerance)

                if current_price > allowed_with_tol:
                    deviation_pct = float(
                        (current_price - allowed) / allowed * 100
                    )
                    return {
                        "contracted_rate": float(base_rate),
                        "allowed_rate": float(allowed),
                        "current_rate": float(current_price),
                        "deviation_pct": round(deviation_pct, 2),
                        "contract_vendor_id": c_vid,
                    }

        return None

    # ------------------------------------------------------------------
    # Step 4: Volume context
    # ------------------------------------------------------------------

    @staticmethod
    def _check_volume_changes(series: list[dict]) -> Optional[str]:
        if len(series) < 4:
            return None
        mid = len(series) // 2
        first_half = [r["quantity"] for r in series[:mid]]
        second_half = [r["quantity"] for r in series[mid:]]
        avg_first = sum(first_half) / len(first_half) if first_half else 1
        avg_second = sum(second_half) / len(second_half) if second_half else 1

        if avg_first == 0:
            return None
        change_pct = abs(avg_second - avg_first) / avg_first * 100
        if change_pct > 30:
            direction = "increased" if avg_second > avg_first else "decreased"
            return (
                f"Average quantity {direction} by {change_pct:.0f}% "
                f"(from {avg_first:.1f} to {avg_second:.1f}), "
                f"which may partially explain price changes"
            )
        return None

    # ------------------------------------------------------------------
    # Finding emission
    # ------------------------------------------------------------------

    def _emit_findings(
        self,
        vendor_id: str,
        norm_item: str,
        series: list[dict],
        analysis: dict,
        contract_info: Optional[dict],
        volume_note: Optional[str],
    ) -> None:
        cum_6mo_thresh = self.config.get("cumulative_threshold_6mo_pct", 8)
        cum_12mo_thresh = self.config.get("cumulative_threshold_12mo_pct", 12)
        jump_thresh = self.config.get("single_jump_threshold_pct", 5)
        p_sig = self.config.get("p_value_significance", 0.05)

        vendor_name = series[0].get("vendor_name", vendor_id)
        invoice_ids = list({r["invoice_id"] for r in series})
        cum_pct = analysis["cumulative_change_pct"]
        period = analysis["period_months"]
        first_price = analysis["first_price"]
        last_price = analysis["last_price"]

        # Price history for evidence / charting
        price_history = [
            {"date": r["date"].isoformat(), "price": r["unit_price"],
             "quantity": r["quantity"]}
            for r in series
        ]

        # ---- Cumulative drift -----------------------------------------
        cumulative_flagged = False
        if cum_pct > 0:
            threshold_pct = cum_6mo_thresh if period <= 6 else cum_12mo_thresh
            if cum_pct >= threshold_pct and analysis["p_value"] <= p_sig:
                cumulative_flagged = True
                severity = self._severity_from_pct(cum_pct)

                est_annual_qty = sum(r["quantity"] for r in series) / max(period, 1) * 12
                price_increase = last_price - first_price
                amount_at_risk = Decimal(str(round(price_increase * est_annual_qty, 2)))

                evidence: dict = {
                    "pattern": "cumulative_drift",
                    "cumulative_change_pct": cum_pct,
                    "period_months": period,
                    "data_points": analysis["data_points"],
                    "slope": analysis["slope"],
                    "r_squared": analysis["r_squared"],
                    "p_value": analysis["p_value"],
                    "price_history": price_history,
                }
                if volume_note:
                    evidence["volume_note"] = volume_note
                if contract_info:
                    evidence["contract_breach"] = contract_info

                desc_parts = [
                    f"Price increased {cum_pct:.1f}% over {period} months "
                    f"for '{norm_item}' from {vendor_name} "
                    f"(${first_price:.2f} → ${last_price:.2f})"
                ]
                if analysis["is_accelerating"]:
                    desc_parts.append("Trend is accelerating")
                    evidence["is_accelerating"] = True
                if contract_info:
                    desc_parts.append(
                        f"Exceeds contracted rate by "
                        f"{contract_info['deviation_pct']:.1f}%"
                    )

                confidence = 0.85 if analysis["r_squared"] >= 0.5 else 0.70
                self.create_finding(
                    severity=severity,
                    confidence=confidence,
                    vendor_id=vendor_id,
                    vendor_name=vendor_name,
                    invoice_ids=invoice_ids,
                    amount_at_risk=max(amount_at_risk, Decimal("0")),
                    description=". ".join(desc_parts),
                    evidence=evidence,
                    recommended_action=(
                        "Review pricing trend and negotiate rate adjustment"
                    ),
                )

        # ---- Single-jump spike ----------------------------------------
        max_jump = analysis["max_single_jump_pct"]
        if max_jump >= jump_thresh and not cumulative_flagged:
            severity = (
                Severity.CRITICAL if max_jump >= 12
                else Severity.REVIEW if max_jump >= 8
                else Severity.INFORMATIONAL
            )
            jump_date = analysis["max_jump_date"]
            # Find the invoice at that date
            jump_inv_ids = [
                r["invoice_id"] for r in series
                if r["date"] == jump_date
            ] or invoice_ids[:2]

            self.create_finding(
                severity=severity,
                confidence=0.80,
                vendor_id=vendor_id,
                vendor_name=vendor_name,
                invoice_ids=jump_inv_ids,
                amount_at_risk=Decimal(str(round(
                    abs(max_jump / 100 * last_price * sum(
                        r["quantity"] for r in series
                    ) / max(len(series), 1)), 2,
                ))),
                description=(
                    f"Single price jump of {max_jump:.1f}% on "
                    f"{jump_date.isoformat() if jump_date else 'N/A'} "
                    f"for '{norm_item}' from {vendor_name}"
                ),
                evidence={
                    "pattern": "single_jump",
                    "jump_pct": max_jump,
                    "jump_date": jump_date.isoformat() if jump_date else None,
                    "price_history": price_history,
                },
                recommended_action=(
                    "Verify justification for the price increase"
                ),
            )

        # ---- Contract breach (standalone, if not already included) -----
        if contract_info and not cumulative_flagged:
            dev = contract_info["deviation_pct"]
            self.create_finding(
                severity=Severity.CRITICAL,
                confidence=0.90,
                vendor_id=vendor_id,
                vendor_name=vendor_name,
                invoice_ids=[series[-1]["invoice_id"]],
                amount_at_risk=Decimal(str(round(
                    (contract_info["current_rate"] - contract_info["allowed_rate"])
                    * series[-1]["quantity"], 2,
                ))),
                description=(
                    f"Current rate ${contract_info['current_rate']:.2f} exceeds "
                    f"contracted rate ${contract_info['contracted_rate']:.2f} "
                    f"(allowed ${contract_info['allowed_rate']:.2f} with escalation) "
                    f"by {dev:.1f}% for '{norm_item}' from {vendor_name}"
                ),
                evidence={
                    "pattern": "contract_breach",
                    **contract_info,
                    "price_history": price_history,
                },
                recommended_action=(
                    "Request rate correction per contract terms"
                ),
            )

    @staticmethod
    def _severity_from_pct(cumulative_pct: float) -> Severity:
        if cumulative_pct >= 15:
            return Severity.CRITICAL
        if cumulative_pct >= 8:
            return Severity.REVIEW
        return Severity.INFORMATIONAL
