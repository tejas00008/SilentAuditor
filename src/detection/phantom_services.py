"""Phantom service and unmatched-PO detection.

Five detection methods:

1. **PO matching gap** — invoice lines without a matching PO, or
   partially matched invoices where the unmatched portion exceeds a
   threshold.
2. **Description vagueness** — line items whose text is too generic to
   verify (e.g. "miscellaneous", "consulting services").
3. **Vendor-service category mismatch** — a vendor billing for services
   outside its established category profile.
4. **One-off charge** — line-item categories that appear exactly once
   for a vendor, combined with vagueness and missing PO.
5. **Delivery / receipt cross-reference** — invoiced goods with no
   matching goods-receipt record.

Methods 1 and 5 require supplementary PO or receipt data; when those
are absent the module runs in degraded mode using only Methods 2–4.
"""

import logging
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from typing import Optional

import pandas as pd

from src.detection.base_detector import BaseDetector
from src.utils.constants import Finding, ModuleName, Severity
from src.utils.date_utils import days_between, parse_date
from src.utils.similarity import normalize_text

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


def _to_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


class PhantomServicesDetector(BaseDetector):
    """Detect phantom services, unmatched POs, and category mismatches."""

    def get_module_name(self) -> ModuleName:
        return ModuleName.PHANTOM_SERVICES

    def get_required_fields(self) -> list[str]:
        return ["vendor_id", "line_item_description", "total_amount", "invoice_date"]

    def get_optional_fields(self) -> list[str]:
        return ["po_number", "vendor_category", "quantity", "unit_price"]

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

        receipts_df = supp.get("receipts")
        if isinstance(receipts_df, list):
            receipts_df = pd.DataFrame(receipts_df) if receipts_df else None

        has_po_col = "po_number" in invoice_data.columns
        has_receipts = receipts_df is not None and len(receipts_df) > 0

        # Build vendor category profile from the full invoice history
        vendor_profile = self._build_vendor_profile(invoice_data)

        # Methods that always run
        self._detect_vagueness(invoice_data, has_po_col)
        self._detect_category_mismatch(invoice_data, vendor_profile)
        self._detect_one_off_charges(invoice_data, vendor_profile, has_po_col)

        # Methods requiring supplementary data
        if has_po_col:
            self._detect_po_gaps(invoice_data)

        if has_receipts:
            self._detect_receipt_gaps(invoice_data, receipts_df)

        # --- Contract scope awareness ---
        # If contracts exist and a line item matches scope, suppress
        # category_mismatch and one_off findings for that vendor.
        contracts = supp.get("contracts", [])
        if contracts:
            self._suppress_contract_covered(contracts)

        # --- Intra-module deduplication ---
        # Group findings by invoice_id, keep highest-confidence per invoice.
        self.findings = self._dedup_findings(self.findings)

        return self.findings

    # ------------------------------------------------------------------
    # Post-detection: contract scope suppression
    # ------------------------------------------------------------------

    def _suppress_contract_covered(self, contracts: list) -> None:
        """Suppress category_mismatch and one_off findings when a contract
        covers the vendor's scope of work."""
        # Build set of vendor_ids that have contracts
        contracted_vids: set[str] = set()
        for c in contracts:
            vid = c.vendor_id if hasattr(c, "vendor_id") else str(c.get("vendor_id", ""))
            if vid:
                contracted_vids.add(vid)

        for f in self.findings:
            if f.suppressed:
                continue
            method = f.evidence.get("method", "")
            if method in ("category_mismatch", "one_off_charge") and f.vendor_id in contracted_vids:
                f.suppressed = True
                f.suppression_reason = "Vendor has active contract covering scope"

    # ------------------------------------------------------------------
    # Post-detection: intra-module deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_findings(findings: list[Finding]) -> list[Finding]:
        """Deduplicate and apply multi-signal convergence filter.

        1. Group findings by invoice_id.
        2. Keep only the highest-confidence finding per invoice.
        3. **Suppress single-signal findings** below $10K — one risk
           indicator alone (just "no PO" or just "vague") is insufficient
           evidence for a fraud finding.
        """
        # Amount threshold: single-signal findings below this are suppressed
        SINGLE_SIGNAL_MIN_AMOUNT = 10_000

        # Group by invoice_id
        by_invoice: dict[str, list[Finding]] = defaultdict(list)
        no_invoice: list[Finding] = []

        for f in findings:
            if f.suppressed:
                no_invoice.append(f)  # keep suppressed as-is
                continue
            inv_ids = f.invoice_ids
            if inv_ids:
                key = inv_ids[0]
                by_invoice[key].append(f)
            else:
                no_invoice.append(f)

        deduped: list[Finding] = list(no_invoice)

        for inv_id, group in by_invoice.items():
            # Count distinct methods that flagged this invoice
            methods = list({f.evidence.get("method", "") for f in group})
            n_signals = len(methods)

            # Sort by confidence descending, pick the best
            group.sort(key=lambda f: f.confidence, reverse=True)
            survivor = group[0]

            # Merge evidence from sibling findings
            survivor.evidence["corroborating_methods"] = methods
            survivor.evidence["signal_count"] = n_signals
            survivor.evidence["original_finding_count"] = len(group)

            # Multi-signal convergence filter:
            # If only 1 detection method flagged this invoice AND the amount
            # is below the threshold, suppress it — single signals are noise.
            amt = float(survivor.amount_at_risk)
            if n_signals <= 1 and amt < SINGLE_SIGNAL_MIN_AMOUNT:
                survivor.suppressed = True
                survivor.suppression_reason = (
                    f"Single signal ({methods[0] if methods else 'unknown'}) "
                    f"below ${SINGLE_SIGNAL_MIN_AMOUNT:,} threshold"
                )
            elif n_signals >= 2:
                # Boost confidence for multi-signal convergence
                survivor.confidence = min(survivor.confidence + 0.10, 0.95)

            deduped.append(survivor)

        return deduped

    # ------------------------------------------------------------------
    # Vendor category profile
    # ------------------------------------------------------------------

    @staticmethod
    def _build_vendor_profile(df: pd.DataFrame) -> dict[str, Counter]:
        """Build ``{vendor_id: Counter({normalised_desc_keyword: count})}``."""
        profiles: dict[str, Counter] = defaultdict(Counter)
        for _, row in df.iterrows():
            vid = _safe_str(row.get("vendor_id"))
            desc = _safe_str(row.get("line_item_description"))
            if not vid or not desc:
                continue
            norm = normalize_text(desc)
            # Use the first two significant words as a rough category key
            words = [w for w in norm.split() if len(w) >= 3]
            cat_key = " ".join(words[:2]) if words else norm
            if cat_key:
                profiles[vid][cat_key] += 1
        return dict(profiles)

    # ==================================================================
    # Method 1 — PO Matching Gap Analysis
    # ==================================================================

    def _detect_po_gaps(self, df: pd.DataFrame) -> None:
        min_amount = _to_float(self.config.get("unmatched_po_min_amount", 1000))
        partial_pct = _to_float(self.config.get("partial_match_deviation_pct", 20))
        self.stats["tier1_calls"] += 1

        # Build vendor PO history: if vendor routinely has no POs, suppress
        vendor_po_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "no_po": 0})
        for _, row in df.iterrows():
            vid = _safe_str(row.get("vendor_id"))
            if vid:
                vendor_po_stats[vid]["total"] += 1
                if not _safe_str(row.get("po_number")):
                    vendor_po_stats[vid]["no_po"] += 1

        for _, row in df.iterrows():
            po = _safe_str(row.get("po_number"))
            amt = _to_float(row.get("total_amount", 0))
            desc = _safe_str(row.get("line_item_description"))
            inv_id = _safe_str(row.get("invoice_id", row.get("invoice_number", "")))
            vid = _safe_str(row.get("vendor_id"))
            vname = _safe_str(row.get("vendor_name", vid))

            if amt < min_amount:
                continue

            # Skip vendors with 10+ invoices where PO-less billing is the norm
            stats = vendor_po_stats.get(vid, {})
            if stats.get("total", 0) >= 10 and stats.get("no_po", 0) >= stats.get("total", 1) * 0.8:
                continue

            if not po:
                self.create_finding(
                    severity=Severity.REVIEW,
                    confidence=0.75,
                    vendor_id=vid,
                    vendor_name=vname,
                    invoice_ids=[inv_id],
                    amount_at_risk=_to_decimal(amt),
                    description=(
                        f"Invoice line from {vname} for ${amt:,.2f} "
                        f"has no matching PO: \"{desc[:80]}\""
                    ),
                    evidence={
                        "method": "po_gap",
                        "subtype": "no_po",
                        "amount": amt,
                        "description": desc[:200],
                    },
                    recommended_action=(
                        "Verify that goods/services were ordered and received"
                    ),
                )

    # ==================================================================
    # Method 2 — Description Vagueness Scoring
    # ==================================================================

    def _detect_vagueness(self, df: pd.DataFrame, has_po: bool) -> None:
        threshold = int(self.config.get("vagueness_flag_threshold", 25))
        amount_floor = _to_float(self.config.get("vagueness_min_amount", 2000))
        self.stats["tier1_calls"] += 1

        for _, row in df.iterrows():
            desc = _safe_str(row.get("line_item_description"))
            if not desc:
                continue

            score = self.llm_client.score_description_vagueness(desc)

            po = _safe_str(row.get("po_number")) if has_po else ""
            amt = _to_float(row.get("total_amount", 0))
            inv_id = _safe_str(row.get("invoice_id", row.get("invoice_number", "")))
            vid = _safe_str(row.get("vendor_id"))
            vname = _safe_str(row.get("vendor_name", vid))

            if score < threshold and not po and amt >= amount_floor:
                severity = Severity.CRITICAL if score < 20 else Severity.REVIEW
                self.create_finding(
                    severity=severity,
                    confidence=0.80 if score < 20 else 0.70,
                    vendor_id=vid,
                    vendor_name=vname,
                    invoice_ids=[inv_id],
                    amount_at_risk=_to_decimal(amt),
                    description=(
                        f"Vague description (specificity {score}/100) with no PO "
                        f"from {vname}: \"{desc[:80]}\" — ${amt:,.2f}"
                    ),
                    evidence={
                        "method": "vagueness",
                        "specificity_score": score,
                        "description": desc[:200],
                        "has_po": False,
                    },
                    recommended_action=(
                        "Request detailed description and supporting documentation"
                    ),
                )

    # ==================================================================
    # Method 3 — Vendor-Service Category Mismatch
    # ==================================================================

    def _detect_category_mismatch(
        self,
        df: pd.DataFrame,
        vendor_profile: dict[str, Counter],
    ) -> None:
        cat_min = _to_float(self.config.get("category_mismatch_min_amount", 3000))
        self.stats["tier1_calls"] += 1

        for _, row in df.iterrows():
            vid = _safe_str(row.get("vendor_id"))
            desc = _safe_str(row.get("line_item_description"))
            amt = _to_float(row.get("total_amount", 0))

            if not vid or not desc or amt < cat_min:
                continue

            profile = vendor_profile.get(vid)
            if not profile or sum(profile.values()) < 5:
                continue  # not enough history (raised from 3 to 5)

            norm = normalize_text(desc)
            words = [w for w in norm.split() if len(w) >= 3]
            cat_key = " ".join(words[:2]) if words else norm

            total_items = sum(profile.values())
            cat_count = profile.get(cat_key, 0)

            inv_id = _safe_str(row.get("invoice_id", row.get("invoice_number", "")))
            vname = _safe_str(row.get("vendor_name", vid))

            if cat_count <= 1 and (cat_count / total_items) < 0.01:
                # Never-seen category AND represents <1% of vendor history
                self.create_finding(
                    severity=Severity.REVIEW,
                    confidence=0.75,
                    vendor_id=vid,
                    vendor_name=vname,
                    invoice_ids=[inv_id],
                    amount_at_risk=_to_decimal(amt),
                    description=(
                        f"Category mismatch: {vname} has never billed for "
                        f"\"{cat_key}\" before — ${amt:,.2f}"
                    ),
                    evidence={
                        "method": "category_mismatch",
                        "subtype": "never_seen",
                        "category_key": cat_key,
                        "vendor_total_items": total_items,
                        "description": desc[:200],
                    },
                    recommended_action=(
                        "Verify vendor is qualified for this type of work"
                    ),
                )
            elif (cat_count / total_items) < 0.01 and amt >= 5000:
                # Very rare category (< 2% of historical billing)
                self.create_finding(
                    severity=Severity.INFORMATIONAL,
                    confidence=0.60,
                    vendor_id=vid,
                    vendor_name=vname,
                    invoice_ids=[inv_id],
                    amount_at_risk=_to_decimal(amt),
                    description=(
                        f"Rare category for {vname}: \"{cat_key}\" represents "
                        f"only {cat_count}/{total_items} prior items — ${amt:,.2f}"
                    ),
                    evidence={
                        "method": "category_mismatch",
                        "subtype": "rare_category",
                        "category_key": cat_key,
                        "category_count": cat_count,
                        "vendor_total_items": total_items,
                        "pct_of_history": round(cat_count / total_items * 100, 2),
                    },
                    recommended_action=(
                        "Review whether this service is within vendor's scope"
                    ),
                )

    # ==================================================================
    # Method 4 — One-Off Charge Detection
    # ==================================================================

    def _detect_one_off_charges(
        self,
        df: pd.DataFrame,
        vendor_profile: dict[str, Counter],
        has_po: bool,
    ) -> None:
        one_off_min = _to_float(self.config.get("one_off_min_amount", 2000))
        vagueness_thresh = int(self.config.get("vagueness_flag_threshold", 40))
        self.stats["tier1_calls"] += 1

        for _, row in df.iterrows():
            vid = _safe_str(row.get("vendor_id"))
            desc = _safe_str(row.get("line_item_description"))
            amt = _to_float(row.get("total_amount", 0))

            if not vid or not desc or amt < one_off_min:
                continue

            profile = vendor_profile.get(vid)
            if not profile or sum(profile.values()) < 5:
                continue

            norm = normalize_text(desc)
            words = [w for w in norm.split() if len(w) >= 3]
            cat_key = " ".join(words[:2]) if words else norm
            cat_count = profile.get(cat_key, 0)

            if cat_count != 1:
                continue  # not a one-off

            # Risk scoring
            vagueness = self.llm_client.score_description_vagueness(desc)
            po = _safe_str(row.get("po_number")) if has_po else ""
            risk_score = 0
            if vagueness < vagueness_thresh:
                risk_score += 2
            if not po:
                risk_score += 1
            if amt >= 5000:
                risk_score += 1

            if risk_score < 2:
                continue

            inv_id = _safe_str(row.get("invoice_id", row.get("invoice_number", "")))
            vname = _safe_str(row.get("vendor_name", vid))
            severity = Severity.REVIEW if risk_score >= 3 else Severity.INFORMATIONAL

            self.create_finding(
                severity=severity,
                confidence=0.65,
                vendor_id=vid,
                vendor_name=vname,
                invoice_ids=[inv_id],
                amount_at_risk=_to_decimal(amt),
                description=(
                    f"One-off charge from {vname}: \"{desc[:80]}\" — ${amt:,.2f} "
                    f"(risk score {risk_score}/4, specificity {vagueness}/100)"
                ),
                evidence={
                    "method": "one_off_charge",
                    "risk_score": risk_score,
                    "specificity_score": vagueness,
                    "has_po": bool(po),
                    "category_key": cat_key,
                    "description": desc[:200],
                },
                recommended_action=(
                    "Investigate one-time charge outside vendor's normal scope"
                ),
            )

    # ==================================================================
    # Method 5 — Delivery / Receipt Cross-Reference
    # ==================================================================

    # Service types that typically don't have goods receipts
    _SERVICE_KEYWORDS = frozenset([
        "consulting", "legal", "staffing", "advisory", "professional",
        "training", "audit", "design", "engineering", "management",
        "inspection", "supervision", "labor", "hr", "recruiting",
    ])

    def _detect_receipt_gaps(
        self, df: pd.DataFrame, receipts_df: pd.DataFrame,
    ) -> None:
        tolerance_days = int(self.config.get("delivery_date_tolerance_days", 7))
        self.stats["tier1_calls"] += 1

        # Build receipt lookup: {invoice_id: [receipt_records]}
        receipt_lookup: dict[str, list[dict]] = defaultdict(list)
        id_col = "invoice_id" if "invoice_id" in receipts_df.columns else "invoice_number"
        for _, rrow in receipts_df.iterrows():
            rid = _safe_str(rrow.get(id_col))
            if rid:
                receipt_lookup[rid].append({
                    "receipt_id": _safe_str(rrow.get("receipt_id")),
                    "receipt_date": _to_date(rrow.get("receipt_date")),
                    "status": _safe_str(rrow.get("status")),
                    "received_by": _safe_str(rrow.get("received_by")),
                })

        # Calculate dataset receipt coverage — if < 50% have receipts,
        # the dataset simply doesn't track receipts well, so disable method
        inv_id_col = "invoice_id" if "invoice_id" in df.columns else "invoice_number"
        unique_inv_ids = df[inv_id_col].dropna().unique()
        total_unique = len(unique_inv_ids)
        matched = sum(1 for iid in unique_inv_ids if str(iid).strip() in receipt_lookup)
        coverage = matched / total_unique if total_unique > 0 else 0.0

        if coverage < 0.50:
            logger.warning(
                "Receipt coverage %.0f%% (< 50%%) — disabling receipt gap method "
                "to avoid false positives", coverage * 100,
            )
            return

        # Check each invoice line against receipts
        checked_ids: set[str] = set()

        for _, row in df.iterrows():
            inv_id = _safe_str(row.get(inv_id_col))
            if not inv_id or inv_id in checked_ids:
                continue
            checked_ids.add(inv_id)

            vid = _safe_str(row.get("vendor_id"))
            vname = _safe_str(row.get("vendor_name", vid))
            amt = _to_float(row.get("total_amount", 0))
            inv_date = _to_date(row.get("invoice_date"))
            desc = _safe_str(row.get("line_item_description"))

            # Skip service-type invoices that don't need goods receipts
            desc_lower = desc.lower()
            if any(kw in desc_lower for kw in self._SERVICE_KEYWORDS):
                continue

            receipts = receipt_lookup.get(inv_id, [])

            if not receipts and amt > 0:
                self.create_finding(
                    severity=Severity.REVIEW,
                    confidence=0.80,
                    vendor_id=vid,
                    vendor_name=vname,
                    invoice_ids=[inv_id],
                    amount_at_risk=_to_decimal(amt),
                    description=(
                        f"No goods receipt for invoice {inv_id} from "
                        f"{vname} — ${amt:,.2f}: \"{desc[:60]}\""
                    ),
                    evidence={
                        "method": "receipt_gap",
                        "subtype": "no_receipt",
                        "description": desc[:200],
                    },
                    recommended_action=(
                        "Confirm that goods/services were actually received"
                    ),
                )
