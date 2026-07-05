"""Contract compliance verification against invoice charges.

Six checks:

1. **Rate compliance** — invoiced rates vs contracted rates with
   escalation.
2. **Payment term compliance** — missed discounts, late payments.
3. **Volume commitment** — unapplied volume discounts.
4. **Scope boundary** — line items outside contracted scope.
5. **Contract expiration** — post-expiry billing and renewal alerts.
6. **Escalation clause** — rate increases exceeding the contract formula.

This module **cannot** run without contract data.
"""

import logging
import re
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz

from src.detection.base_detector import BaseDetector
from src.utils.constants import (
    ContractTerms,
    Finding,
    ModuleName,
    Severity,
)
from src.utils.date_utils import days_between, months_between, parse_date
from src.utils.similarity import normalize_text

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Payment-term parser
# ------------------------------------------------------------------

_TERM_PATTERN = re.compile(
    r"(?:(\d+)\s*/\s*(\d+)\s+)?Net\s+(\d+)",
    re.I,
)


def parse_payment_terms(terms: str) -> dict:
    """Parse common payment-term strings.

    Returns ``{net_days, discount_pct, discount_days}`` or
    ``{net_days: 0}`` for unrecognised terms.

    Handles: ``Net 30``, ``2/10 Net 30``, ``1/15 Net 45``,
    ``Due on Receipt``, ``COD``.
    """
    if not terms:
        return {"net_days": 0, "discount_pct": 0, "discount_days": 0}

    lower = terms.lower().strip()
    if lower in ("due on receipt", "cod", "cash on delivery"):
        return {"net_days": 0, "discount_pct": 0, "discount_days": 0}

    m = _TERM_PATTERN.search(terms)
    if m:
        disc_pct = int(m.group(1)) if m.group(1) else 0
        disc_days = int(m.group(2)) if m.group(2) else 0
        net_days = int(m.group(3))
        return {
            "net_days": net_days,
            "discount_pct": disc_pct,
            "discount_days": disc_days,
        }

    return {"net_days": 0, "discount_pct": 0, "discount_days": 0}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_date(val) -> Optional[date]:
    if isinstance(val, date):
        return val
    return parse_date(str(val)) if val is not None else None


def _to_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _safe(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


class ContractComplianceDetector(BaseDetector):
    """Verify invoices against contracted terms."""

    def get_module_name(self) -> ModuleName:
        return ModuleName.CONTRACT_COMPLIANCE

    def get_required_fields(self) -> list[str]:
        return ["vendor_id", "total_amount", "invoice_date",
                "line_item_description", "unit_price"]

    def get_optional_fields(self) -> list[str]:
        return ["payment_date", "quantity"]

    # Override can_run: contracts must be provided
    def can_run(self, available_fields: list[str]) -> tuple[bool, float]:
        can, eff = super().can_run(available_fields)
        if not can:
            return False, 0.0
        # The detect() method checks for contracts in supplementary_data;
        # here we can only validate column presence.
        return True, eff

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

        contracts = self._load_contracts(supplementary_data)
        if not contracts:
            self.logger.warning(
                "No contract data available — module cannot run. "
                "supplementary_data keys: %s",
                list(supplementary_data.keys()) if supplementary_data else "None",
            )
            return self.findings

        self.logger.info(
            "Contract compliance running with %d contracts", len(contracts),
        )

        # Index contracts by vendor_id
        contract_map: dict[str, ContractTerms] = {}
        for c in contracts:
            vid = c.vendor_id if isinstance(c, ContractTerms) else str(c.get("vendor_id", ""))
            if vid:
                contract_map[vid] = c

        inv_groups = self._group_invoices(invoice_data)

        for vid, contract in contract_map.items():
            invs = inv_groups.get(vid, [])
            if not invs:
                continue

            self._check_rates(vid, invs, contract)
            self._check_payment_terms(vid, invs, contract)
            self._check_volume(vid, invs, contract)
            self._check_scope(vid, invs, contract)
            self._check_expiration(vid, invs, contract)
            self._check_escalation(vid, invs, contract)

        self.stats["tier1_calls"] += 6
        return self.findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_contracts(supp: Optional[dict]) -> list:
        if not supp:
            return []
        raw = supp.get("contracts", [])
        if not raw:
            return []
        return raw

    @staticmethod
    def _group_invoices(df: pd.DataFrame) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = defaultdict(list)
        id_col = "invoice_id" if "invoice_id" in df.columns else "invoice_number"
        for _, row in df.iterrows():
            vid = _safe(row.get("vendor_id"))
            if not vid:
                continue
            groups[vid].append({
                "invoice_id": _safe(row.get(id_col)),
                "vendor_name": _safe(row.get("vendor_name", vid)),
                "invoice_date": _to_date(row.get("invoice_date")),
                "total_amount": _to_float(row.get("total_amount")),
                "unit_price": _to_float(row.get("unit_price")),
                "quantity": _to_float(row.get("quantity", 1)),
                "payment_date": _to_date(row.get("payment_date")),
                "description": _safe(row.get("line_item_description")),
            })
        return dict(groups)

    def _contract_attr(self, contract: object, attr: str, default: object = None) -> object:
        if isinstance(contract, ContractTerms):
            return getattr(contract, attr, default)
        return contract.get(attr, default)

    def _contract_rates(self, contract) -> list[dict]:
        raw = self._contract_attr(contract, "rates", [])
        result = []
        for r in raw:
            if isinstance(r, dict):
                result.append(r)
            else:
                result.append({
                    "item_description": getattr(r, "item_description", ""),
                    "unit": getattr(r, "unit", ""),
                    "rate": getattr(r, "rate", Decimal("0")),
                })
        return result

    def _contract_discounts(self, contract) -> list[dict]:
        raw = self._contract_attr(contract, "volume_discounts", [])
        result = []
        for vd in raw:
            if isinstance(vd, dict):
                result.append(vd)
            else:
                result.append({
                    "threshold_quantity": getattr(vd, "threshold_quantity", 0),
                    "discount_pct": getattr(vd, "discount_pct", Decimal("0")),
                })
        return result

    # ==================================================================
    # Check 1 — Rate Compliance
    # ==================================================================

    def _check_rates(self, vid: str, invs: list[dict], contract) -> None:
        tol_pct = _to_float(self.config.get("rate_tolerance_pct", 3))
        rates = self._contract_rates(contract)
        if not rates:
            return

        escalation = _to_float(self._contract_attr(contract, "annual_escalation_pct", 0)) / 100
        start = self._contract_attr(contract, "contract_start_date")
        if isinstance(start, str):
            start = _to_date(start)

        for inv in invs:
            price = inv["unit_price"]
            if price <= 0:
                continue
            desc = inv["description"]
            norm_desc = normalize_text(desc)

            for rate_entry in rates:
                rate_desc = normalize_text(str(rate_entry.get("item_description", "")))
                base_rate = _to_float(rate_entry.get("rate", 0))
                if base_rate <= 0:
                    continue

                # Match description
                sim = fuzz.token_sort_ratio(norm_desc, rate_desc) / 100.0
                if sim < 0.40:
                    continue

                # Compute allowed rate with escalation
                inv_date = inv["invoice_date"] or date.today()
                years = months_between(start or date(2024, 1, 1), inv_date) / 12 if start else 0
                allowed = base_rate * (1 + escalation) ** max(years, 0)
                allowed_with_tol = allowed * (1 + tol_pct / 100)

                if price > allowed_with_tol:
                    deviation = (price - allowed) / allowed * 100
                    vname = inv["vendor_name"]
                    self.create_finding(
                        severity=Severity.CRITICAL if deviation > 10 else Severity.REVIEW,
                        confidence=0.85,
                        vendor_id=vid,
                        vendor_name=vname,
                        invoice_ids=[inv["invoice_id"]],
                        amount_at_risk=Decimal(str(round(
                            (price - allowed) * inv["quantity"], 2,
                        ))),
                        description=(
                            f"Rate overcharge: {vname} billed "
                            f"${price:.2f}/unit vs allowed "
                            f"${allowed:.2f}/unit ({deviation:+.1f}%)"
                        ),
                        evidence={
                            "check": "rate_compliance",
                            "invoiced_rate": price,
                            "contracted_rate": base_rate,
                            "allowed_rate": round(allowed, 2),
                            "deviation_pct": round(deviation, 2),
                            "escalation_applied": round(escalation * 100, 1),
                        },
                        recommended_action="Request rate correction per contract",
                    )
                    break  # one finding per invoice

    # ==================================================================
    # Check 2 — Payment Term Compliance
    # ==================================================================

    def _check_payment_terms(self, vid: str, invs: list[dict], contract) -> None:
        terms_str = str(self._contract_attr(contract, "payment_terms", ""))
        terms = parse_payment_terms(terms_str)
        if terms["discount_pct"] == 0:
            return  # no early-payment discount to check

        buffer = int(self.config.get("discount_window_buffer_days", 1))
        disc_days = terms["discount_days"]
        disc_pct = terms["discount_pct"]

        for inv in invs:
            inv_date = inv["invoice_date"]
            pay_date = inv["payment_date"]
            if not inv_date or not pay_date:
                continue

            days_to_pay = days_between(inv_date, pay_date)
            if days_to_pay <= disc_days + buffer:
                # Paid within discount window — check if discount was applied
                expected_discount = inv["total_amount"] * disc_pct / 100
                if expected_discount > 1:
                    vname = inv["vendor_name"]
                    self.create_finding(
                        severity=Severity.REVIEW,
                        confidence=0.80,
                        vendor_id=vid,
                        vendor_name=vname,
                        invoice_ids=[inv["invoice_id"]],
                        amount_at_risk=Decimal(str(round(expected_discount, 2))),
                        description=(
                            f"Missed early-payment discount: {vname} paid "
                            f"in {days_to_pay} days (within {disc_pct}/{disc_days} "
                            f"window) — ${expected_discount:,.2f} discount not applied"
                        ),
                        evidence={
                            "check": "payment_terms",
                            "subtype": "missed_discount",
                            "payment_terms": terms_str,
                            "days_to_pay": days_to_pay,
                            "discount_window": disc_days,
                            "discount_pct": disc_pct,
                            "expected_discount": round(expected_discount, 2),
                        },
                        recommended_action="Claim early-payment discount retroactively",
                    )

    # ==================================================================
    # Check 3 — Volume Commitment
    # ==================================================================

    def _check_volume(self, vid: str, invs: list[dict], contract) -> None:
        discounts = self._contract_discounts(contract)
        if not discounts:
            return

        total_qty = sum(inv["quantity"] for inv in invs)
        vname = invs[0]["vendor_name"] if invs else vid

        for vd in discounts:
            threshold = _to_float(vd.get("threshold_quantity", 0))
            disc_pct = _to_float(vd.get("discount_pct", 0))
            if threshold <= 0 or disc_pct <= 0:
                continue

            if total_qty >= threshold:
                total_spend = sum(inv["total_amount"] for inv in invs)
                missed = total_spend * disc_pct / 100
                inv_ids = [inv["invoice_id"] for inv in invs[:10]]
                self.create_finding(
                    severity=Severity.REVIEW,
                    confidence=0.80,
                    vendor_id=vid,
                    vendor_name=vname,
                    invoice_ids=inv_ids,
                    amount_at_risk=Decimal(str(round(missed, 2))),
                    description=(
                        f"Volume discount not applied: {total_qty:.0f} units "
                        f"purchased (threshold {threshold:.0f}), {disc_pct}% "
                        f"discount = ${missed:,.2f} savings"
                    ),
                    evidence={
                        "check": "volume_commitment",
                        "total_quantity": total_qty,
                        "threshold": threshold,
                        "discount_pct": disc_pct,
                        "missed_savings": round(missed, 2),
                    },
                    recommended_action="Apply volume discount retroactively",
                )

    # ==================================================================
    # Check 4 — Scope Boundary
    # ==================================================================

    def _check_scope(self, vid: str, invs: list[dict], contract) -> None:
        scope = self._contract_attr(contract, "scope_of_work", [])
        if not scope:
            return

        sim_threshold = _to_float(self.config.get("scope_similarity_threshold", 0.70))
        norm_scope = [normalize_text(s) for s in scope]
        vname = invs[0]["vendor_name"] if invs else vid

        for inv in invs:
            desc = inv["description"]
            if not desc:
                continue
            norm_desc = normalize_text(desc)

            best_sim = max(
                (fuzz.token_sort_ratio(norm_desc, ns) / 100.0 for ns in norm_scope),
                default=0.0,
            )

            if best_sim < sim_threshold and inv["total_amount"] >= 1000:
                self.create_finding(
                    severity=Severity.REVIEW,
                    confidence=0.70,
                    vendor_id=vid,
                    vendor_name=vname,
                    invoice_ids=[inv["invoice_id"]],
                    amount_at_risk=Decimal(str(round(inv["total_amount"], 2))),
                    description=(
                        f"Out-of-scope billing: \"{desc[:60]}\" from {vname} "
                        f"doesn't match contract scope (best similarity "
                        f"{best_sim:.0%})"
                    ),
                    evidence={
                        "check": "scope_boundary",
                        "description": desc[:200],
                        "best_scope_similarity": round(best_sim, 3),
                        "scope_items": scope[:5],
                    },
                    recommended_action="Verify whether service is within contract scope",
                )

    # ==================================================================
    # Check 5 — Contract Expiration
    # ==================================================================

    def _check_expiration(self, vid: str, invs: list[dict], contract) -> None:
        end_date = self._contract_attr(contract, "contract_end_date")
        if isinstance(end_date, str):
            end_date = _to_date(end_date)
        if not end_date:
            return

        renewal_days = int(self.config.get("renewal_alert_days", 90))
        vname = invs[0]["vendor_name"] if invs else vid

        # Collect post-expiry invoices, then create at most ONE finding
        # per vendor summarizing the exposure (instead of one per invoice).
        post_expiry_invoices = []
        for inv in invs:
            inv_date = inv["invoice_date"]
            if not inv_date:
                continue
            if inv_date > end_date:
                days_past = days_between(end_date, inv_date)
                post_expiry_invoices.append({
                    **inv,
                    "days_past": days_past,
                })

        if not post_expiry_invoices:
            return

        # Only flag as actionable finding if gap > 90 days AND total
        # post-expiry spend is significant. Short gaps (<=90 days) are
        # normal contract renewal delays in construction.
        max_days = max(i["days_past"] for i in post_expiry_invoices)
        total_post_expiry = sum(i["total_amount"] for i in post_expiry_invoices)
        inv_ids = [i["invoice_id"] for i in post_expiry_invoices[:10]]

        if max_days <= 90:
            # Short gap — informational only, likely renewal delay
            self.create_finding(
                severity=Severity.INFORMATIONAL,
                confidence=0.40,  # Below confidence gate — will be filtered
                vendor_id=vid,
                vendor_name=vname,
                invoice_ids=inv_ids,
                amount_at_risk=Decimal(str(round(total_post_expiry, 2))),
                description=(
                    f"Post-expiry billing: {vname} has {len(post_expiry_invoices)} "
                    f"invoices totaling ${total_post_expiry:,.0f} after contract "
                    f"expired {end_date.isoformat()} (max gap: {max_days} days)"
                ),
                evidence={
                    "check": "expiration",
                    "subtype": "post_expiry_short",
                    "contract_end": end_date.isoformat(),
                    "max_days_past_expiry": max_days,
                    "invoice_count": len(post_expiry_invoices),
                    "total_amount": total_post_expiry,
                },
                recommended_action="Confirm contract renewal is in progress",
            )
        else:
            # Long gap — create a single consolidated finding
            self.create_finding(
                severity=Severity.REVIEW,
                confidence=0.70,
                vendor_id=vid,
                vendor_name=vname,
                invoice_ids=inv_ids,
                amount_at_risk=Decimal(str(round(total_post_expiry, 2))),
                description=(
                    f"Post-expiry billing: {vname} has {len(post_expiry_invoices)} "
                    f"invoices totaling ${total_post_expiry:,.0f} after contract "
                    f"expired {end_date.isoformat()} (max gap: {max_days} days)"
                ),
                evidence={
                    "check": "expiration",
                    "subtype": "post_expiry_long",
                    "contract_end": end_date.isoformat(),
                    "max_days_past_expiry": max_days,
                    "invoice_count": len(post_expiry_invoices),
                    "total_amount": total_post_expiry,
                },
                recommended_action="Renegotiate contract or cease payments",
            )

    # ==================================================================
    # Check 6 — Escalation Clause
    # ==================================================================

    def _check_escalation(self, vid: str, invs: list[dict], contract) -> None:
        max_esc = _to_float(self._contract_attr(contract, "annual_escalation_pct", 0))
        if max_esc <= 0:
            return

        rates = self._contract_rates(contract)
        if not rates:
            return

        start = self._contract_attr(contract, "contract_start_date")
        if isinstance(start, str):
            start = _to_date(start)

        vname = invs[0]["vendor_name"] if invs else vid

        for inv in invs:
            price = inv["unit_price"]
            if price <= 0:
                continue
            inv_date = inv["invoice_date"] or date.today()
            desc = inv["description"]
            norm_desc = normalize_text(desc)

            for rate_entry in rates:
                rate_desc = normalize_text(str(rate_entry.get("item_description", "")))
                base_rate = _to_float(rate_entry.get("rate", 0))
                if base_rate <= 0:
                    continue

                sim = fuzz.token_sort_ratio(norm_desc, rate_desc) / 100.0
                if sim < 0.40:
                    continue

                years = months_between(start or date(2024, 1, 1), inv_date) / 12 if start else 0
                max_allowed = base_rate * (1 + max_esc / 100) ** max(years, 0)
                actual_esc_pct = (price - base_rate) / base_rate * 100 if base_rate else 0

                if price > max_allowed * 1.01 and actual_esc_pct > max_esc:
                    excess_pct = actual_esc_pct - max_esc
                    self.create_finding(
                        severity=Severity.REVIEW,
                        confidence=0.85,
                        vendor_id=vid,
                        vendor_name=vname,
                        invoice_ids=[inv["invoice_id"]],
                        amount_at_risk=Decimal(str(round(
                            (price - max_allowed) * inv["quantity"], 2,
                        ))),
                        description=(
                            f"Excess escalation: {vname} applied "
                            f"{actual_esc_pct:.1f}% increase vs max "
                            f"{max_esc:.1f}% allowed"
                        ),
                        evidence={
                            "check": "escalation",
                            "base_rate": base_rate,
                            "invoiced_rate": price,
                            "max_allowed_rate": round(max_allowed, 2),
                            "actual_escalation_pct": round(actual_esc_pct, 1),
                            "max_escalation_pct": max_esc,
                            "excess_pct": round(excess_pct, 1),
                        },
                        recommended_action="Request rate correction per escalation clause",
                    )
                    break
