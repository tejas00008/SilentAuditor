"""Data quality assessment for uploaded invoice data.

Evaluates field completeness, data type validity, date-range depth, and
determines per-module readiness so the user knows exactly what each
detection module can (and cannot) do with the data they provided.
"""

import logging
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional

import pandas as pd

from src.utils.constants import DataQualityReport, ModuleName
from src.utils.date_utils import parse_date

logger = logging.getLogger(__name__)

# Fields we inspect during the assessment.  For each field we record
# whether it is present in the DataFrame, and what percentage of rows
# contain a non-empty value.
_EXPECTED_FIELDS: list[str] = [
    "invoice_number",
    "vendor_name",
    "vendor_id",
    "invoice_date",
    "total_amount",
    "line_item_description",
    "unit_price",
    "quantity",
    "po_number",
    "payment_date",
    "approved_by",
]

# Module weights used for the overall readiness score.
_MODULE_WEIGHTS: dict[str, float] = {
    ModuleName.DUPLICATE_DETECTION.value: 0.20,
    ModuleName.PRICE_CREEP.value:         0.20,
    ModuleName.PHANTOM_SERVICES.value:    0.12,
    ModuleName.VENDOR_COLLUSION.value:    0.08,
    ModuleName.CONTRACT_COMPLIANCE.value: 0.08,
    ModuleName.VENDOR_BEHAVIOR.value:     0.12,
    ModuleName.MARKET_PRICE.value:        0.08,
    ModuleName.SPLIT_INVOICING.value:     0.12,
}


def _col_present(df: pd.DataFrame, col: str) -> bool:
    """Return ``True`` if *col* exists in *df*."""
    return col in df.columns


def _completeness(df: pd.DataFrame, col: str) -> float:
    """Fraction of rows with a non-empty value (0.0–1.0).

    Returns 0.0 when the column is absent.
    """
    if col not in df.columns or len(df) == 0:
        return 0.0
    non_empty = df[col].apply(
        lambda v: v is not None and str(v).strip() != ""
    ).sum()
    return non_empty / len(df)


def _months_of_data(df: pd.DataFrame) -> int:
    """Estimate how many calendar months the invoice date range spans."""
    if "invoice_date" not in df.columns or len(df) == 0:
        return 0
    dates: list[date] = []
    for v in df["invoice_date"]:
        if isinstance(v, date):
            dates.append(v)
        elif v is not None:
            parsed = parse_date(str(v))
            if parsed is not None:
                dates.append(parsed)
    if len(dates) < 2:
        return 0
    earliest = min(dates)
    latest = max(dates)
    return (latest.year - earliest.year) * 12 + (latest.month - earliest.month)


class DataQualityAssessor:
    """Assess uploaded data and determine per-module readiness."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        invoice_df: pd.DataFrame,
        vendor_df: Optional[pd.DataFrame] = None,
        contracts: Optional[list] = None,
        receipts_df: Optional[pd.DataFrame] = None,
        approval_thresholds: Optional[list] = None,
    ) -> DataQualityReport:
        """Run the full data-quality assessment.

        Returns a :class:`DataQualityReport` summarising statistics,
        field completeness, module readiness, issues, and an overall
        readiness score.
        """
        # -- 1. Basic statistics ----------------------------------------
        total_invoices = len(invoice_df)

        date_range = self._compute_date_range(invoice_df)
        months = _months_of_data(invoice_df)

        unique_vendors = self._count_unique_vendors(invoice_df)

        # -- 2. Field completeness --------------------------------------
        fields_present: dict[str, dict] = {}
        for field in _EXPECTED_FIELDS:
            present = _col_present(invoice_df, field)
            pct = _completeness(invoice_df, field) if present else 0.0
            fields_present[field] = {
                "present": present,
                "completeness": round(pct, 3),
            }

        # -- 3. Data-type validation (collect issues) -------------------
        issues: list[str] = []
        issues.extend(self._validate_types(invoice_df, fields_present))

        # -- 4. Module readiness ----------------------------------------
        module_readiness = self._assess_modules(
            invoice_df, vendor_df, contracts, receipts_df,
            approval_thresholds, fields_present, months, issues,
        )

        # -- 5. Additional issues from supplementary data ---------------
        if not contracts or len(contracts) == 0:
            issues.append(
                "No contract data provided (Module 5 — Contract Compliance cannot run)"
            )
        else:
            logger.info(
                "Data quality: %d contracts available for compliance checks",
                len(contracts),
            )
        if vendor_df is None or len(vendor_df) == 0:
            issues.append(
                "No vendor master data (Module 4 — Vendor Collusion has reduced effectiveness)"
            )

        # -- 6. Overall readiness score ---------------------------------
        overall = self._compute_overall_score(module_readiness)

        return DataQualityReport(
            total_invoices=total_invoices,
            date_range=date_range,
            unique_vendors=unique_vendors,
            fields_present=fields_present,
            module_readiness=module_readiness,
            data_quality_issues=issues,
            overall_readiness_score=round(overall, 2),
        )

    def print_report(self, report: DataQualityReport) -> None:
        """Pretty-print the quality report to the logger."""
        logger.info("=" * 60)
        logger.info("  SilentAuditor — Data Quality Report")
        logger.info("=" * 60)
        logger.info(
            "  Invoices: %d  |  Vendors: %d  |  Date range: %s → %s",
            report.total_invoices,
            report.unique_vendors,
            report.date_range.get("earliest", "N/A"),
            report.date_range.get("latest", "N/A"),
        )
        logger.info("-" * 60)

        # Field completeness
        logger.info("  Field Completeness:")
        for field, info in report.fields_present.items():
            status = "✓" if info["present"] else "✗"
            pct = f"{info['completeness'] * 100:.0f}%" if info["present"] else "—"
            logger.info("    %s %-25s %s", status, field, pct)

        logger.info("-" * 60)

        # Module readiness
        logger.info("  Module Readiness:")
        for module, info in report.module_readiness.items():
            run = "CAN RUN" if info["can_run"] else "BLOCKED"
            eff = f"{info['effectiveness'] * 100:.0f}%"
            logger.info("    [%s] %-25s eff=%s", run, module, eff)
            if info.get("notes"):
                logger.info("           ↳ %s", info["notes"])

        logger.info("-" * 60)

        # Issues
        if report.data_quality_issues:
            logger.info("  Issues (%d):", len(report.data_quality_issues))
            for issue in report.data_quality_issues:
                logger.info("    • %s", issue)
        else:
            logger.info("  No data quality issues found.")

        logger.info("-" * 60)

        # Overall score
        score = report.overall_readiness_score
        if score >= 0.70:
            label = "GOOD"
        elif score >= 0.50:
            label = "FAIR"
        else:
            label = "POOR"
        logger.info("  Overall Readiness: %.0f%% (%s)", score * 100, label)
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_date_range(df: pd.DataFrame) -> dict:
        if "invoice_date" not in df.columns or len(df) == 0:
            return {"earliest": None, "latest": None, "months": 0}

        dates: list[date] = []
        for v in df["invoice_date"]:
            if isinstance(v, date):
                dates.append(v)
            elif v is not None:
                parsed = parse_date(str(v))
                if parsed is not None:
                    dates.append(parsed)
        if not dates:
            return {"earliest": None, "latest": None, "months": 0}

        earliest = min(dates)
        latest = max(dates)
        months = (latest.year - earliest.year) * 12 + (latest.month - earliest.month)
        return {
            "earliest": earliest.isoformat(),
            "latest": latest.isoformat(),
            "months": months,
        }

    @staticmethod
    def _count_unique_vendors(df: pd.DataFrame) -> int:
        for col in ("vendor_name", "vendor_id"):
            if col in df.columns:
                return int(df[col].nunique())
        return 0

    @staticmethod
    def _validate_types(
        df: pd.DataFrame, fields_present: dict,
    ) -> list[str]:
        """Validate data types and return a list of issue strings."""
        issues: list[str] = []
        n = len(df)
        if n == 0:
            issues.append("Dataset is empty — no invoices to analyse")
            return issues

        # Date fields
        for date_field in ("invoice_date", "payment_date"):
            if not fields_present.get(date_field, {}).get("present"):
                continue
            unparseable = 0
            for v in df[date_field]:
                if v is None or str(v).strip() == "":
                    continue
                if isinstance(v, date):
                    continue
                if parse_date(str(v)) is None:
                    unparseable += 1
            if unparseable > 0:
                pct = unparseable / n * 100
                issues.append(
                    f"{pct:.0f}% of {date_field} values are unparseable "
                    f"({unparseable}/{n} rows)"
                )

        # Numeric fields
        for num_field in ("total_amount", "unit_price", "quantity"):
            if not fields_present.get(num_field, {}).get("present"):
                continue
            non_numeric = 0
            negative = 0
            for v in df[num_field]:
                if v is None or str(v).strip() == "":
                    continue
                s = str(v).replace("$", "").replace(",", "").replace("€", "").replace("£", "")
                if s.startswith("(") and s.endswith(")"):
                    s = s[1:-1]
                try:
                    d = Decimal(s)
                    if d < 0:
                        negative += 1
                except InvalidOperation:
                    non_numeric += 1
            if non_numeric > 0:
                pct = non_numeric / n * 100
                issues.append(
                    f"{pct:.0f}% of {num_field} values are non-numeric "
                    f"({non_numeric}/{n} rows)"
                )
            if negative > 0:
                pct = negative / n * 100
                issues.append(
                    f"{pct:.0f}% of {num_field} values are negative "
                    f"({negative}/{n} rows)"
                )

        # Nulls in required fields
        for req in ("invoice_number", "vendor_name", "invoice_date", "total_amount"):
            info = fields_present.get(req, {})
            if not info.get("present"):
                issues.append(f"Required field '{req}' is missing entirely")
            elif info["completeness"] < 1.0:
                missing_pct = (1 - info["completeness"]) * 100
                issues.append(
                    f"{missing_pct:.0f}% of '{req}' values are empty"
                )

        return issues

    # ------------------------------------------------------------------
    # Module readiness
    # ------------------------------------------------------------------

    def _assess_modules(
        self,
        invoice_df: pd.DataFrame,
        vendor_df: Optional[pd.DataFrame],
        contracts: Optional[list],
        receipts_df: Optional[pd.DataFrame],
        approval_thresholds: Optional[list],
        fp: dict,
        months: int,
        issues: list[str],
    ) -> dict:
        """Build per-module readiness dicts."""
        has = lambda f: fp.get(f, {}).get("present", False)  # noqa: E731
        comp = lambda f: fp.get(f, {}).get("completeness", 0.0)  # noqa: E731
        has_vendor_master = vendor_df is not None and len(vendor_df) > 0
        has_contracts = contracts is not None and len(contracts) > 0
        has_receipts = receipts_df is not None and len(receipts_df) > 0
        has_thresholds = approval_thresholds is not None and len(approval_thresholds) > 0
        n = len(invoice_df)

        readiness: dict = {}

        # ---- Module 1: Duplicate Detection ----------------------------
        can_run = (
            has("invoice_number") and has("invoice_date")
            and has("total_amount")
            and (has("vendor_name") or has("vendor_id"))
        )
        eff = 0.0
        notes_parts: list[str] = []
        if can_run:
            eff = 0.95 if has("line_item_description") else 0.75
            if has("po_number"):
                eff = min(eff + 0.05, 1.0)
            if not has("line_item_description"):
                notes_parts.append("Line item descriptions would improve matching")
        else:
            notes_parts.append("Requires invoice_number, vendor, date, and amount")
        readiness[ModuleName.DUPLICATE_DETECTION.value] = {
            "can_run": can_run, "effectiveness": round(eff, 2),
            "notes": "; ".join(notes_parts) if notes_parts else None,
        }

        # ---- Module 2: Price Creep ------------------------------------
        can_run = (
            has("unit_price") and has("line_item_description")
            and has("invoice_date")
            and (has("vendor_name") or has("vendor_id"))
        )
        eff = 0.0
        notes_parts = []
        if can_run:
            eff = 1.0 if has_contracts else 0.80
            if comp("unit_price") < 0.5:
                coverage_pct = comp("unit_price") * 100
                eff *= comp("unit_price") / 0.5  # proportional degradation
                eff = max(eff, 0.30)
                issues.append(
                    f"{100 - coverage_pct:.0f}% of unit prices missing "
                    f"(Module 2 will have reduced coverage)"
                )
                notes_parts.append("Low unit_price coverage")
        else:
            notes_parts.append("Requires unit_price, line_item_description, date, and vendor")
        readiness[ModuleName.PRICE_CREEP.value] = {
            "can_run": can_run, "effectiveness": round(eff, 2),
            "notes": "; ".join(notes_parts) if notes_parts else None,
        }

        # ---- Module 3: Phantom Services -------------------------------
        can_run = (
            has("total_amount") and has("line_item_description")
            and (has("vendor_name") or has("vendor_id"))
        )
        eff = 0.0
        notes_parts = []
        if can_run:
            if has_receipts:
                eff = 0.95
            elif has("po_number"):
                eff = 0.90
            else:
                eff = 0.55
                issues.append(
                    "No PO data provided (Module 3 — Phantom Services running in limited mode)"
                )
                notes_parts.append("PO or receipt data would significantly improve detection")
            desc_comp = comp("line_item_description")
            if desc_comp < 0.85 and desc_comp > 0:
                missing_pct = (1 - desc_comp) * 100
                issues.append(
                    f"{missing_pct:.0f}% of line items have no description "
                    f"(affects Modules 3, 7)"
                )
        else:
            notes_parts.append("Requires vendor, line_item_description, and amount")
        readiness[ModuleName.PHANTOM_SERVICES.value] = {
            "can_run": can_run, "effectiveness": round(eff, 2),
            "notes": "; ".join(notes_parts) if notes_parts else None,
        }

        # ---- Module 4: Vendor Collusion -------------------------------
        can_run = (
            has("total_amount") and has("invoice_date")
            and (has("vendor_name") or has("vendor_id"))
        )
        eff = 0.0
        notes_parts = []
        if can_run:
            if has_vendor_master and has("approved_by"):
                eff = 0.90
            elif has_vendor_master:
                eff = 0.70
            else:
                eff = 0.40
                notes_parts.append("Vendor master + approver data would improve detection")
        else:
            notes_parts.append("Requires vendor, amount, and date")
        readiness[ModuleName.VENDOR_COLLUSION.value] = {
            "can_run": can_run, "effectiveness": round(eff, 2),
            "notes": "; ".join(notes_parts) if notes_parts else None,
        }

        # ---- Module 5: Contract Compliance ----------------------------
        can_run = has_contracts
        eff = 0.0
        notes_parts = []
        if can_run:
            eff = 1.0
        else:
            notes_parts.append("Requires contract data to run")
        readiness[ModuleName.CONTRACT_COMPLIANCE.value] = {
            "can_run": can_run, "effectiveness": round(eff, 2),
            "notes": "; ".join(notes_parts) if notes_parts else None,
        }

        # ---- Module 6: Vendor Behavior --------------------------------
        min_months_run = 6
        can_run = (
            has("total_amount") and has("invoice_date")
            and (has("vendor_name") or has("vendor_id"))
            and months >= min_months_run
        )
        eff = 0.0
        notes_parts = []
        if can_run:
            eff = 0.85 if months >= 12 else 0.60
        else:
            if months < min_months_run:
                issues.append(
                    f"Only {months} months of history "
                    f"(Module 6 — Vendor Behavior needs minimum 6 months)"
                )
                notes_parts.append(f"Need ≥ 6 months of data (have {months})")
            else:
                notes_parts.append("Requires vendor, amount, and date")
        readiness[ModuleName.VENDOR_BEHAVIOR.value] = {
            "can_run": can_run, "effectiveness": round(eff, 2),
            "notes": "; ".join(notes_parts) if notes_parts else None,
        }

        # ---- Module 7: Market Price -----------------------------------
        can_run = (
            has("line_item_description") and has("unit_price")
            and (has("vendor_name") or has("vendor_id"))
        )
        eff = 0.0
        notes_parts = []
        if can_run:
            eff = 0.60  # internal-only baseline
            notes_parts.append("Cross-client benchmarks not yet available")
        else:
            notes_parts.append("Requires vendor, line_item_description, and unit_price")
        readiness[ModuleName.MARKET_PRICE.value] = {
            "can_run": can_run, "effectiveness": round(eff, 2),
            "notes": "; ".join(notes_parts) if notes_parts else None,
        }

        # ---- Module 8: Split Invoicing --------------------------------
        can_run = (
            has("total_amount") and has("invoice_date")
            and (has("vendor_name") or has("vendor_id"))
        )
        eff = 0.0
        notes_parts = []
        if can_run:
            if has_thresholds and has("po_number"):
                eff = 0.95
            elif has_thresholds:
                eff = 0.90
            else:
                eff = 0.70
                notes_parts.append("Approval thresholds would improve detection")
        else:
            notes_parts.append("Requires vendor, amount, and date")
        readiness[ModuleName.SPLIT_INVOICING.value] = {
            "can_run": can_run, "effectiveness": round(eff, 2),
            "notes": "; ".join(notes_parts) if notes_parts else None,
        }

        return readiness

    @staticmethod
    def _compute_overall_score(module_readiness: dict) -> float:
        """Weighted average of module effectiveness scores."""
        total_weight = 0.0
        weighted_sum = 0.0
        for module, weight in _MODULE_WEIGHTS.items():
            info = module_readiness.get(module)
            if info is not None:
                weighted_sum += info["effectiveness"] * weight
                total_weight += weight
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight
