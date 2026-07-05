"""Duplicate invoice detection using exact, fuzzy, semantic, and cross-PO matching.

Four detection layers run sequentially:

1. **Exact match** — same invoice number + vendor + amount.
2. **Fuzzy composite** — blocked by vendor within a time window, scored
   on amount, date, line-item text, quantity, and PO similarity.
3. **Semantic (LLM)** — ambiguous pairs from Layer 2 are sent to the LLM
   for a final duplicate/not-duplicate judgement.
4. **Cross-PO** — same vendor, similar amount & date but *different* POs.

After all layers, false-positive suppression removes recurring
subscriptions, progress-billing patterns, and credit/rebill pairs.
"""

import logging
import re
from collections import defaultdict
from decimal import Decimal
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz

from src.detection.base_detector import BaseDetector
from src.utils.constants import Finding, ModuleName, Severity
from src.utils.date_utils import days_between, parse_date

logger = logging.getLogger(__name__)

_PROGRESS_KEYWORDS = re.compile(
    r"\b(phase|milestone|stage|period|installment|progress)\b", re.I,
)
_SEQUENTIAL_PATTERN = re.compile(
    r"\b(\d+)\s*(?:of|/)\s*(\d+)\b"
    r"|\bphase\s*(\d+)\b"
    r"|\bstage\s*(\d+)\b",
    re.I,
)


def _to_date(val: object) -> Optional[date]:
    """Coerce a value to ``datetime.date``, returning ``None`` on failure."""
    from datetime import date as _date
    if isinstance(val, _date):
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


class DuplicateInvoiceDetector(BaseDetector):
    """Detect duplicate invoice payments across four detection layers."""

    def get_module_name(self) -> ModuleName:
        return ModuleName.DUPLICATE_DETECTION

    def get_required_fields(self) -> list[str]:
        return ["invoice_number", "vendor_id", "invoice_date", "total_amount"]

    def get_optional_fields(self) -> list[str]:
        return ["line_item_description", "po_number", "quantity", "unit_price"]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect(
        self,
        invoice_data: pd.DataFrame,
        vendor_data: Optional[pd.DataFrame] = None,
        supplementary_data: Optional[dict] = None,
    ) -> list[Finding]:
        df = invoice_data.copy()
        self.stats["invoices_analyzed"] = len(df)
        self.findings = []
        self._init_flagged_pairs()

        has_line_items = "line_item_description" in df.columns
        has_po = "po_number" in df.columns

        # Layer 1 — runs on raw data before aggregation so that rows
        # sharing the same invoice_number + vendor + amount are detected
        # even when they share the same invoice_id.
        self._detect_exact_duplicates(df)
        self.stats["tier1_calls"] += 1

        # Aggregate to invoice level (one row per invoice_id) for the
        # remaining layers which compare *distinct* invoices.
        inv_df = self._aggregate_to_invoice_level(df)

        # Layer 2
        ambiguous = self._detect_fuzzy_duplicates(inv_df, has_line_items, has_po, df)
        self.stats["tier1_calls"] += 1

        # Layer 3
        if ambiguous:
            self._detect_semantic_duplicates(ambiguous, df)

        # Layer 4
        self._detect_cross_po_duplicates(inv_df, has_line_items, df)
        self.stats["tier1_calls"] += 1

        # False-positive suppression
        self._suppress_false_positives(self.findings, inv_df)

        return self.findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_to_invoice_level(df: pd.DataFrame) -> pd.DataFrame:
        """Collapse line-item rows into one row per invoice."""
        group_cols = [
            c for c in [
                "invoice_id", "invoice_number", "vendor_id", "vendor_name",
                "invoice_date", "total_amount", "po_number", "payment_date",
                "payment_status",
            ]
            if c in df.columns
        ]
        if "invoice_id" not in df.columns:
            if "invoice_number" in df.columns:
                df = df.copy()
                df["invoice_id"] = df["invoice_number"]
            else:
                return df

        agg: dict = {}
        if "line_item_description" in df.columns:
            agg["line_item_description"] = lambda x: " | ".join(
                _safe_str(v) for v in x if _safe_str(v)
            )
        if "quantity" in df.columns:
            agg["quantity"] = list
        if "unit_price" in df.columns:
            agg["unit_price"] = list

        if not agg:
            return df.drop_duplicates(subset=["invoice_id"])

        non_agg = [c for c in group_cols if c in df.columns]
        return (
            df.groupby(non_agg, dropna=False, sort=False)
            .agg(agg)
            .reset_index()
        )

    def _init_flagged_pairs(self) -> None:
        """Initialise the set of already-flagged invoice-pair keys."""
        self._flagged_pairs: set[tuple[str, str]] = set()

    def _mark_flagged(self, inv_a: str, inv_b: str) -> None:
        key = (min(inv_a, inv_b), max(inv_a, inv_b))
        self._flagged_pairs.add(key)

    def _already_flagged(self, inv_a: str, inv_b: str) -> bool:
        """O(1) check whether this pair is already in findings."""
        key = (min(inv_a, inv_b), max(inv_a, inv_b))
        return key in self._flagged_pairs

    # ==================================================================
    # Layer 1: Exact Match
    # ==================================================================

    def _detect_exact_duplicates(self, df: pd.DataFrame) -> None:
        conf = self.config.get("exact_match_confidence", 0.99)

        # Deduplicate to one row per (invoice_number, vendor_id,
        # total_amount, invoice_date line-item) so that multi-line-item
        # invoices don't inflate the count.  We keep the first line-item
        # row as a representative.
        id_col = "invoice_id" if "invoice_id" in df.columns else "invoice_number"
        deduped = df.drop_duplicates(
            subset=[id_col, "line_item_description"]
            if "line_item_description" in df.columns
            else [id_col],
        )
        # Now group by the key fields and look for groups whose unique
        # invoice-level identifiers appear more than once.
        # We need to count distinct *rows that were originally separate*
        # in the raw file.  Two rows with the same invoice_number +
        # vendor_id + amount are an exact duplicate if they represent
        # separate payment records.
        #
        # Strategy: group by (invoice_number, vendor_id, total_amount)
        # then count how many *distinct line-item rows* exist.  If a
        # single-line invoice appears twice, there are 2 rows.
        groups = df.groupby(
            ["invoice_number", "vendor_id", "total_amount"], dropna=False,
        )
        seen_groups: set[tuple] = set()
        for key, group in groups:
            inv_num, vid, amt = key
            # Count distinct line-item-level rows.  For a genuine exact
            # duplicate injected by the sample-data generator, the same
            # invoice_number appears on two separate CSV rows.
            # Deduplicate by line_item_id (if present) or the full row.
            if "line_item_id" in group.columns:
                unique_li = group["line_item_id"].nunique()
            else:
                unique_li = len(group)

            if unique_li < 2:
                continue
            gkey = (str(inv_num), str(vid), str(amt))
            if gkey in seen_groups:
                continue
            seen_groups.add(gkey)

            inv_ids = group[id_col].unique().tolist()
            vname = _safe_str(group.iloc[0].get("vendor_name", vid))
            self.create_finding(
                severity=Severity.CRITICAL,
                confidence=conf,
                vendor_id=str(vid),
                vendor_name=vname,
                invoice_ids=inv_ids,
                amount_at_risk=_to_decimal(amt),
                description=(
                    f"Exact duplicate: invoice {inv_num} from {vname} "
                    f"for ${_to_decimal(amt):,.2f} appears {unique_li} times"
                ),
                evidence={
                    "layer": "exact_match",
                    "invoice_number": str(inv_num),
                    "count": unique_li,
                },
                recommended_action="Verify whether duplicate payment was made",
            )

    # ==================================================================
    # Layer 2: Fuzzy Composite Match
    # ==================================================================

    def _detect_fuzzy_duplicates(
        self,
        inv_df: pd.DataFrame,
        has_line_items: bool,
        has_po: bool,
        raw_df: pd.DataFrame,
    ) -> list[tuple]:
        threshold = self.config.get("fuzzy_composite_threshold", 0.85)
        ambiguous_lower = self.config.get("ambiguous_zone_lower", 0.70)
        date_window = self.config.get("date_window_days", 90)

        ambiguous_pairs: list[tuple] = []

        # Weights
        if has_line_items:
            w_amt, w_date, w_li, w_qty, w_po = 0.30, 0.15, 0.30, 0.10, 0.15
        else:
            w_amt, w_date, w_li, w_qty, w_po = 0.45, 0.25, 0.0, 0.10, 0.20

        # Block by vendor, sort by date for early-exit optimisation
        vendor_groups = inv_df.groupby("vendor_id", dropna=False)
        for vid, vgroup in vendor_groups:
            rows = vgroup.to_dict("records")
            n = len(rows)
            if n < 2:
                continue

            # Pre-parse dates and sort for O(n log n) windowed comparison
            for r in rows:
                r["_date"] = _to_date(r.get("invoice_date"))
            rows = [r for r in rows if r["_date"] is not None]
            rows.sort(key=lambda r: r["_date"])
            n = len(rows)

            for i in range(n):
                date_i = rows[i]["_date"]
                for j in range(i + 1, n):
                    date_j = rows[j]["_date"]
                    gap = (date_j - date_i).days  # always >= 0 (sorted)
                    if gap > date_window:
                        break  # all subsequent j are even further away

                    inv_a = str(rows[i].get("invoice_id", ""))
                    inv_b = str(rows[j].get("invoice_id", ""))
                    if inv_a == inv_b:
                        continue
                    if self._already_flagged(inv_a, inv_b):
                        continue

                    # Quick amount pre-filter: skip if amounts differ > 20%
                    amt_a = abs(float(_to_decimal(rows[i].get("total_amount", 0))))
                    amt_b = abs(float(_to_decimal(rows[j].get("total_amount", 0))))
                    if max(amt_a, amt_b) > 0:
                        amt_diff = abs(amt_a - amt_b) / max(amt_a, amt_b)
                        if amt_diff > 0.20:
                            continue

                    # --- scores ---
                    s_amt = self._amount_similarity(rows[i], rows[j])
                    s_date = self._date_score(gap)
                    s_li = 0.5
                    if has_line_items and w_li > 0:
                        s_li = self._line_item_score(rows[i], rows[j])
                    s_qty = self._quantity_score(rows[i], rows[j])
                    s_po = 0.5
                    if has_po:
                        s_po = self._po_score(rows[i], rows[j])

                    composite = (
                        w_amt * s_amt
                        + w_date * s_date
                        + w_li * s_li
                        + w_qty * s_qty
                        + w_po * s_po
                    )

                    if composite >= threshold:
                        vname = _safe_str(rows[i].get("vendor_name", vid))
                        amt_a = _to_decimal(rows[i].get("total_amount", 0))
                        self.create_finding(
                            severity=Severity.CRITICAL,
                            confidence=round(min(composite, 0.95), 2),
                            vendor_id=str(vid),
                            vendor_name=vname,
                            invoice_ids=[inv_a, inv_b],
                            amount_at_risk=amt_a,
                            description=(
                                f"Near-duplicate pair: {inv_a} and {inv_b} "
                                f"from {vname} (composite score {composite:.2f})"
                            ),
                            evidence={
                                "layer": "fuzzy_match",
                                "composite_score": round(composite, 3),
                                "amount_score": round(s_amt, 3),
                                "date_score": round(s_date, 3),
                                "line_item_score": round(s_li, 3),
                                "po_score": round(s_po, 3),
                                "days_apart": gap,
                            },
                            recommended_action=(
                                "Compare invoices side-by-side to confirm "
                                "duplicate payment"
                            ),
                        )
                        self._mark_flagged(inv_a, inv_b)
                    elif composite >= ambiguous_lower:
                        ambiguous_pairs.append((rows[i], rows[j], composite))

        return ambiguous_pairs

    # ---- Component scores ---------------------------------------------

    @staticmethod
    def _amount_similarity(a: dict, b: dict) -> float:
        amt_a = abs(float(_to_decimal(a.get("total_amount", 0))))
        amt_b = abs(float(_to_decimal(b.get("total_amount", 0))))
        if max(amt_a, amt_b) == 0:
            return 1.0
        pct_diff = abs(amt_a - amt_b) / max(amt_a, amt_b)
        if pct_diff <= 0.01:
            return 1.0
        if pct_diff <= 0.05:
            return 0.8
        if pct_diff <= 0.10:
            return 0.5
        return 0.0

    @staticmethod
    def _date_score(gap: int) -> float:
        if gap == 0:
            return 1.0
        if gap <= 7:
            return 0.8
        if gap <= 30:
            return 0.5
        if gap <= 90:
            return 0.2
        return 0.0

    def _line_item_score(self, a: dict, b: dict) -> float:
        desc_a = _safe_str(a.get("line_item_description"))
        desc_b = _safe_str(b.get("line_item_description"))
        if not desc_a or not desc_b:
            return 0.5
        try:
            sim = self.llm_client.compute_text_similarity(desc_a, desc_b)
            self.stats["tier2_calls"] += 1
            return sim
        except Exception:
            return fuzz.token_sort_ratio(desc_a, desc_b) / 100.0

    @staticmethod
    def _quantity_score(a: dict, b: dict) -> float:
        qa = a.get("quantity")
        qb = b.get("quantity")
        if qa is None or qb is None:
            return 0.5
        if isinstance(qa, list) and isinstance(qb, list):
            set_a = set(str(q) for q in qa)
            set_b = set(str(q) for q in qb)
        else:
            set_a = {str(qa)}
            set_b = {str(qb)}
        if not set_a and not set_b:
            return 0.5
        union = set_a | set_b
        inter = set_a & set_b
        return len(inter) / len(union) if union else 0.5

    @staticmethod
    def _po_score(a: dict, b: dict) -> float:
        po_a = _safe_str(a.get("po_number"))
        po_b = _safe_str(b.get("po_number"))
        if not po_a or not po_b:
            return 0.5
        ratio = fuzz.ratio(po_a, po_b)
        if ratio >= 90:
            return 1.0
        return 0.0  # both exist but different — counter-signal

    # ==================================================================
    # Layer 3: Semantic Duplicate Check
    # ==================================================================

    def _detect_semantic_duplicates(
        self, ambiguous_pairs: list[tuple], raw_df: pd.DataFrame,
    ) -> None:
        for row_a, row_b, composite in ambiguous_pairs:
            inv_a_dict = {
                "vendor_name": _safe_str(row_a.get("vendor_name")),
                "date": _safe_str(row_a.get("invoice_date")),
                "amount": str(row_a.get("total_amount", 0)),
                "line_items": _safe_str(row_a.get("line_item_description")),
                "po_number": _safe_str(row_a.get("po_number")),
            }
            inv_b_dict = {
                "vendor_name": _safe_str(row_b.get("vendor_name")),
                "date": _safe_str(row_b.get("invoice_date")),
                "amount": str(row_b.get("total_amount", 0)),
                "line_items": _safe_str(row_b.get("line_item_description")),
                "po_number": _safe_str(row_b.get("po_number")),
            }
            try:
                result = self.llm_client.assess_duplicate_pair(
                    inv_a_dict, inv_b_dict,
                    customer_id="default",
                )
                self.stats["tier3_calls"] += 1
            except Exception:
                self.logger.debug("LLM call failed for ambiguous pair; skipping")
                continue

            if result.get("is_duplicate") and result.get("confidence", 0) >= 0.70:
                inv_a_id = str(row_a.get("invoice_id", ""))
                inv_b_id = str(row_b.get("invoice_id", ""))
                vid = str(row_a.get("vendor_id", ""))
                vname = _safe_str(row_a.get("vendor_name"))
                amt = _to_decimal(row_a.get("total_amount", 0))
                self.create_finding(
                    severity=Severity.REVIEW,
                    confidence=round(result["confidence"], 2),
                    vendor_id=vid,
                    vendor_name=vname,
                    invoice_ids=[inv_a_id, inv_b_id],
                    amount_at_risk=amt,
                    description=(
                        f"LLM-confirmed duplicate: {inv_a_id} and {inv_b_id} "
                        f"from {vname} — {result.get('reasoning', '')}"
                    ),
                    evidence={
                        "layer": "semantic_llm",
                        "llm_confidence": result["confidence"],
                        "llm_reasoning": result.get("reasoning", ""),
                        "composite_score": round(composite, 3),
                    },
                    recommended_action="Review LLM reasoning and confirm",
                )

    # ==================================================================
    # Layer 4: Cross-PO Duplicate Check
    # ==================================================================

    def _detect_cross_po_duplicates(
        self,
        inv_df: pd.DataFrame,
        has_line_items: bool,
        raw_df: pd.DataFrame,
    ) -> None:
        if "po_number" not in inv_df.columns:
            return

        amt_tol = self.config.get("amount_tolerance_pct", 0.05)
        vendor_groups = inv_df.groupby("vendor_id", dropna=False)

        for vid, vgroup in vendor_groups:
            rows = vgroup.to_dict("records")
            n = len(rows)
            for i in range(n):
                po_i = _safe_str(rows[i].get("po_number"))
                if not po_i:
                    continue
                date_i = _to_date(rows[i].get("invoice_date"))
                if date_i is None:
                    continue
                amt_i = float(_to_decimal(rows[i].get("total_amount", 0)))

                for j in range(i + 1, n):
                    po_j = _safe_str(rows[j].get("po_number"))
                    if not po_j or po_j == po_i:
                        continue
                    date_j = _to_date(rows[j].get("invoice_date"))
                    if date_j is None:
                        continue
                    gap = days_between(date_i, date_j)
                    if gap > 30:
                        continue

                    amt_j = float(_to_decimal(rows[j].get("total_amount", 0)))
                    if max(amt_i, amt_j) == 0:
                        continue
                    pct_diff = abs(amt_i - amt_j) / max(amt_i, amt_j)
                    if pct_diff > amt_tol * 2:  # 10% for cross-PO
                        continue

                    inv_a = str(rows[i].get("invoice_id", ""))
                    inv_b = str(rows[j].get("invoice_id", ""))
                    if self._already_flagged(inv_a, inv_b):
                        continue

                    # Line-item similarity check
                    li_sim = 0.5
                    if has_line_items:
                        li_sim = self._line_item_score(rows[i], rows[j])

                    if li_sim >= 0.85:
                        vname = _safe_str(rows[i].get("vendor_name", vid))
                        self.create_finding(
                            severity=Severity.REVIEW,
                            confidence=0.70,
                            vendor_id=str(vid),
                            vendor_name=vname,
                            invoice_ids=[inv_a, inv_b],
                            amount_at_risk=_to_decimal(rows[i].get("total_amount", 0)),
                            description=(
                                f"Cross-PO duplicate: {inv_a} (PO {po_i}) and "
                                f"{inv_b} (PO {po_j}) from {vname} — "
                                f"similar work under different POs"
                            ),
                            evidence={
                                "layer": "cross_po",
                                "po_a": po_i,
                                "po_b": po_j,
                                "amount_diff_pct": round(pct_diff * 100, 2),
                                "line_item_similarity": round(li_sim, 3),
                                "days_apart": gap,
                            },
                            recommended_action=(
                                "Verify that work billed under both POs was "
                                "actually performed separately"
                            ),
                        )
                        self._mark_flagged(inv_a, inv_b)

    # ==================================================================
    # False Positive Suppression
    # ==================================================================

    def _suppress_false_positives(
        self, findings: list[Finding], inv_df: pd.DataFrame,
    ) -> None:
        recurring_ids = self._detect_recurring_invoices(inv_df)
        progress_ids = self._detect_progress_billing(inv_df)
        credit_ids = self._detect_credit_rebills(inv_df)

        suppression_set = recurring_ids | progress_ids | credit_ids

        for f in findings:
            if f.suppressed:
                continue
            flagged = set(f.invoice_ids)
            overlap = flagged & suppression_set
            if overlap:
                reason_parts = []
                if overlap & recurring_ids:
                    reason_parts.append("recurring subscription pattern")
                if overlap & progress_ids:
                    reason_parts.append("progress billing pattern")
                if overlap & credit_ids:
                    reason_parts.append("credit/rebill pair")
                f.suppressed = True
                f.suppression_reason = "False positive: " + ", ".join(reason_parts)

    def _detect_recurring_invoices(self, inv_df: pd.DataFrame) -> set[str]:
        """Identify recurring subscription invoices that look like duplicates."""
        min_count = self.config.get("recurring_invoice_min_count", 3)
        interval_tol = self.config.get("recurring_interval_tolerance_days", 5)
        recurring_ids: set[str] = set()

        vendor_groups = inv_df.groupby("vendor_id", dropna=False)
        for vid, vgroup in vendor_groups:
            if len(vgroup) < min_count:
                continue

            # Group by approximate amount (within 2%)
            rows = vgroup.to_dict("records")
            rows.sort(key=lambda r: str(r.get("invoice_date", "")))

            amount_groups: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                amt = float(_to_decimal(r.get("total_amount", 0)))
                # Bucket key: round to nearest 100 for grouping
                bucket = str(round(amt, -2)) if amt > 0 else "0"
                amount_groups[bucket].append(r)

            for bucket, grp in amount_groups.items():
                if len(grp) < min_count:
                    continue

                # Check if amounts are within 2% of each other
                amts = [float(_to_decimal(r.get("total_amount", 0))) for r in grp]
                avg_amt = sum(amts) / len(amts)
                if avg_amt == 0:
                    continue
                if any(abs(a - avg_amt) / avg_amt > 0.02 for a in amts):
                    continue

                # Check interval regularity
                dates = []
                for r in grp:
                    d = _to_date(r.get("invoice_date"))
                    if d:
                        dates.append(d)
                dates.sort()
                if len(dates) < min_count:
                    continue

                intervals = [
                    (dates[k + 1] - dates[k]).days
                    for k in range(len(dates) - 1)
                ]
                if not intervals:
                    continue
                avg_interval = sum(intervals) / len(intervals)
                if avg_interval < 20:  # too frequent to be monthly
                    continue
                regular = all(
                    abs(iv - avg_interval) <= interval_tol
                    for iv in intervals
                )
                if regular:
                    for r in grp:
                        recurring_ids.add(str(r.get("invoice_id", "")))

        return recurring_ids

    def _detect_progress_billing(self, inv_df: pd.DataFrame) -> set[str]:
        """Identify progress-billing / milestone invoices."""
        progress_ids: set[str] = set()
        if "line_item_description" not in inv_df.columns:
            return progress_ids

        for _, row in inv_df.iterrows():
            desc = _safe_str(row.get("line_item_description"))
            if _PROGRESS_KEYWORDS.search(desc) or _SEQUENTIAL_PATTERN.search(desc):
                progress_ids.add(str(row.get("invoice_id", "")))

        return progress_ids

    @staticmethod
    def _detect_credit_rebills(inv_df: pd.DataFrame) -> set[str]:
        """Identify credit memo / rebill pairs."""
        credit_ids: set[str] = set()
        if "total_amount" not in inv_df.columns:
            return credit_ids

        # Collect negative-amount invoices
        negatives: list[dict] = []
        positives: list[dict] = []
        for _, row in inv_df.iterrows():
            amt = float(_to_decimal(row.get("total_amount", 0)))
            rec = {
                "invoice_id": str(row.get("invoice_id", "")),
                "vendor_id": str(row.get("vendor_id", "")),
                "amount": amt,
            }
            if amt < 0:
                negatives.append(rec)
            else:
                positives.append(rec)

        for neg in negatives:
            for pos in positives:
                if neg["vendor_id"] != pos["vendor_id"]:
                    continue
                if abs(abs(neg["amount"]) - pos["amount"]) / max(pos["amount"], 1) < 0.01:
                    credit_ids.add(neg["invoice_id"])
                    credit_ids.add(pos["invoice_id"])

        return credit_ids
