"""Standardize invoice monetary amounts for consistent comparison.

Handles multiple currency formats, detects tax components, identifies
credit memos, and validates totals.  All monetary values are stored as
``Decimal`` with two decimal places.
"""

import logging
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_TWO_PLACES = Decimal("0.01")

# Common tax-related keywords found in line-item descriptions.
_TAX_KEYWORDS: set[str] = {
    "tax", "gst", "vat", "hst", "pst", "qst",
    "sales tax", "excise", "duty", "levy",
}

# Standard tax rates to probe when no explicit tax line is present.
_COMMON_TAX_RATES: list[Decimal] = [
    Decimal("0.05"), Decimal("0.07"), Decimal("0.075"),
    Decimal("0.08"), Decimal("0.0875"), Decimal("0.10"),
    Decimal("0.13"), Decimal("0.18"), Decimal("0.20"),
]

_CREDIT_PATTERN = re.compile(r"\b(cr|credit|credit\s*memo|refund)\b", re.I)

# Unreasonably large for mid-market ($5M–$100M revenue).
_MAX_REASONABLE_AMOUNT = Decimal("10_000_000")


def _parse_amount(value: object) -> Optional[Decimal]:
    """Parse a raw value into a two-decimal-place ``Decimal``.

    Handles:
    * Currency symbols: ``$``, ``€``, ``£``, ``₹``
    * Thousand separators: US ``1,234.56`` and EU ``1.234,56``
    * Parenthesised negatives: ``(1234.56)``
    * Trailing CR: ``1234.56 CR``
    * Whitespace and leading/trailing junk
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    # Detect credit suffix
    is_credit = bool(_CREDIT_PATTERN.search(s))
    s = _CREDIT_PATTERN.sub("", s).strip()

    # Strip currency symbols
    s = re.sub(r"[$€£₹]", "", s).strip()

    # Parenthesised negative
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
        is_credit = True

    # Detect EU format: digits.digits,digits (period as thousand sep,
    # comma as decimal).  e.g. "1.234,56"
    # Require the comma-decimal part to positively identify EU format.
    if re.match(r"^-?\d{1,3}(\.\d{3})+,\d{1,2}$", s):
        s = s.replace(".", "").replace(",", ".")
    else:
        # US / standard: comma as thousand separator
        s = s.replace(",", "")

    try:
        d = Decimal(s).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
        if is_credit and d > 0:
            d = -d
        return d
    except InvalidOperation:
        return None


def _is_tax_description(desc: str) -> bool:
    """Return ``True`` if *desc* looks like a tax line item."""
    if not desc:
        return False
    lower = desc.lower()
    return any(kw in lower for kw in _TAX_KEYWORDS)


class AmountNormalizer:
    """Standardize monetary amounts across an invoice DataFrame."""

    def normalize_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all monetary amounts.

        Adds/updates columns:

        * ``gross_amount`` — total charged (always positive or zero)
        * ``net_amount``   — gross minus tax
        * ``tax_amount``   — detected or inferred tax component
        * ``is_credit``    — ``True`` for credit memos / negative amounts
        * ``amount_issues`` — list of validation warning strings per row
        """
        result = df.copy()

        # ---- Step 1+2: parse raw amounts --------------------------------
        result["gross_amount"] = result["total_amount"].apply(_parse_amount)

        # ---- Step 3: tax detection & separation -------------------------
        tax_amounts, net_amounts = self._separate_tax(result)
        result["tax_amount"] = tax_amounts
        result["net_amount"] = net_amounts

        # ---- Step 4: credit detection -----------------------------------
        result["is_credit"] = result.apply(self._detect_credit, axis=1)

        # ---- Step 5: validation -----------------------------------------
        result["amount_issues"] = result.apply(self._validate_row, axis=1)

        return result

    # ------------------------------------------------------------------
    # Tax separation
    # ------------------------------------------------------------------

    def _separate_tax(
        self, df: pd.DataFrame,
    ) -> tuple[list[Optional[Decimal]], list[Optional[Decimal]]]:
        """Split gross amount into net + tax."""
        has_desc = "line_item_description" in df.columns
        tax_col: list[Optional[Decimal]] = []
        net_col: list[Optional[Decimal]] = []

        # Check whether the dataset has any explicit tax lines
        explicit_tax_total = Decimal("0")
        if has_desc:
            for _, row in df.iterrows():
                desc = str(row.get("line_item_description", ""))
                if _is_tax_description(desc):
                    amt = row.get("gross_amount")
                    if isinstance(amt, Decimal):
                        explicit_tax_total += abs(amt)

        for _, row in df.iterrows():
            gross = row.get("gross_amount")
            if not isinstance(gross, Decimal):
                tax_col.append(None)
                net_col.append(None)
                continue

            abs_gross = abs(gross)
            desc = str(row.get("line_item_description", "")) if has_desc else ""

            # If this row IS a tax line, the tax is the whole amount
            if has_desc and _is_tax_description(desc):
                tax_col.append(abs_gross)
                net_col.append(Decimal("0.00"))
                continue

            # If explicit tax lines exist elsewhere, assume this row is net
            if explicit_tax_total > 0:
                tax_col.append(Decimal("0.00"))
                net_col.append(abs_gross)
                continue

            # Probe common tax rates
            inferred_tax = self._infer_tax(abs_gross)
            tax_col.append(inferred_tax)
            net_col.append(
                (abs_gross - inferred_tax).quantize(_TWO_PLACES)
            )

        return tax_col, net_col

    @staticmethod
    def _infer_tax(gross: Decimal) -> Decimal:
        """Try to infer tax from a gross amount using common rates.

        If ``gross / (1 + rate)`` yields a round-dollar net amount
        (within 1 cent) for a common rate, assume that rate was embedded.
        Otherwise return zero.
        """
        for rate in _COMMON_TAX_RATES:
            net = (gross / (1 + rate)).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
            recalc = (net * (1 + rate)).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
            # Check if the amount round-trips cleanly
            if abs(recalc - gross) <= Decimal("0.01"):
                # Extra heuristic: the net should be a "round" number
                # (ends in .00 or .50 or .25 or .75 — reasonable for an
                # invoice).  This avoids false positives on random amounts.
                cents = net % 1
                if cents in (Decimal("0.00"), Decimal("0.25"),
                             Decimal("0.50"), Decimal("0.75")):
                    return (gross - net).quantize(_TWO_PLACES)
        return Decimal("0.00")

    # ------------------------------------------------------------------
    # Credit detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_credit(row: pd.Series) -> bool:
        gross = row.get("gross_amount")
        if isinstance(gross, Decimal) and gross < 0:
            return True
        for col in ("total_amount", "line_item_description", "payment_status"):
            val = str(row.get(col, ""))
            if _CREDIT_PATTERN.search(val):
                return True
        return False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_row(row: pd.Series) -> list[str]:
        issues: list[str] = []
        gross = row.get("gross_amount")
        if gross is None:
            issues.append("Unparseable amount")
            return issues
        if gross == 0:
            issues.append("Zero amount")
        if abs(gross) > _MAX_REASONABLE_AMOUNT:
            issues.append(f"Unusually large amount: {gross}")
        return issues

    # ------------------------------------------------------------------
    # Line-item total reconciliation (call after all rows normalised)
    # ------------------------------------------------------------------

    @staticmethod
    def reconcile_line_items(
        invoice_total: Decimal,
        line_item_amounts: list[Decimal],
        tolerance_pct: float = 1.0,
    ) -> dict:
        """Check whether line-item amounts sum to the invoice total.

        Returns ``{matches, total, line_sum, difference, difference_pct}``.
        """
        line_sum = sum(line_item_amounts)
        diff = abs(invoice_total - line_sum)
        if invoice_total != 0:
            diff_pct = float(diff / abs(invoice_total)) * 100
        else:
            diff_pct = 0.0 if line_sum == 0 else 100.0
        return {
            "matches": diff_pct <= tolerance_pct,
            "total": invoice_total,
            "line_sum": line_sum,
            "difference": diff,
            "difference_pct": round(diff_pct, 2),
        }
