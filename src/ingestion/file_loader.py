"""File loading utilities for CSV, Excel, JSON, and TSV invoice data."""

import json
import logging
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.constants import (
    ApprovalThreshold,
    ContractRate,
    ContractTerms,
    VolumeDiscount,
)
from src.utils.date_utils import parse_date

logger = logging.getLogger(__name__)

_ENCODINGS: list[str] = ["utf-8", "latin-1", "cp1252"]
_CSV_DELIMITERS: list[str] = [",", ";", "\t", "|"]


class FileLoader:
    """Load invoice, vendor, contract, and receipt data from files."""

    # ------------------------------------------------------------------
    # Generic loader
    # ------------------------------------------------------------------

    def load(self, filepath: str) -> pd.DataFrame:
        """Auto-detect file type by extension and load into a DataFrame.

        Supported extensions: ``.csv``, ``.tsv``, ``.xlsx``, ``.xls``,
        ``.json``.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = path.suffix.lower()

        if ext in (".csv", ".tsv"):
            df = self._load_csv(path)
        elif ext in (".xlsx", ".xls"):
            df = self._load_excel(path)
        elif ext == ".json":
            df = self._load_json(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        df = self._clean_strings(df)
        logger.info(
            "Loaded %s: %d rows, %d columns",
            path.name, len(df), len(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    # Format-specific loaders
    # ------------------------------------------------------------------

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Try multiple encodings and delimiters to load a CSV/TSV."""
        last_exc: Optional[Exception] = None
        for encoding in _ENCODINGS:
            for delimiter in _CSV_DELIMITERS:
                try:
                    df = pd.read_csv(
                        path,
                        encoding=encoding,
                        delimiter=delimiter,
                        dtype=str,
                        keep_default_na=False,
                    )
                    # Heuristic: a successful parse should produce > 1 column
                    # (unless the file genuinely has one column).
                    if len(df.columns) >= 2 or delimiter == _CSV_DELIMITERS[-1]:
                        logger.debug(
                            "CSV parsed with encoding=%s delimiter=%r",
                            encoding, delimiter,
                        )
                        return df
                except Exception as exc:
                    last_exc = exc
        raise ValueError(
            f"Unable to parse CSV file {path.name}"
        ) from last_exc

    def _load_excel(self, path: Path) -> pd.DataFrame:
        """Load the first sheet of an Excel workbook."""
        df = pd.read_excel(
            path,
            engine="openpyxl",
            dtype=str,
            keep_default_na=False,
        )
        return df

    def _load_json(self, path: Path) -> pd.DataFrame:
        """Load JSON — handles array-of-objects and nested structures."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, list):
            return pd.DataFrame(data).astype(str)

        # Nested: look for the first key whose value is a list of dicts.
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    logger.debug("Using JSON key %r as record array", key)
                    return pd.DataFrame(value).astype(str)
            # Single flat object → one-row DataFrame
            return pd.DataFrame([data]).astype(str)

        raise ValueError(f"Unexpected JSON structure in {path.name}")

    # ------------------------------------------------------------------
    # String cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing whitespace from all string columns."""
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            df[col] = df[col].str.strip()
        # Also clean column headers
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        return df

    # ------------------------------------------------------------------
    # Domain-specific loaders
    # ------------------------------------------------------------------

    def load_vendor_master(self, filepath: str) -> pd.DataFrame:
        """Load vendor master data (CSV / Excel / JSON)."""
        return self.load(filepath)

    def load_contracts(self, filepath: str) -> list[ContractTerms]:
        """Load contract terms from a JSON file.

        Expected structure: a JSON array of contract objects.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

        records = raw if isinstance(raw, list) else [raw]
        contracts: list[ContractTerms] = []

        for rec in records:
            # Support both canonical field names and alternate formats
            # Rates: accept "rates" (canonical) or "contracted_items" (alternate)
            raw_rates = rec.get("rates", rec.get("contracted_items", []))
            rates = []
            for r in raw_rates:
                # "item_description" (canonical) or "description" (alternate)
                desc = r.get("item_description", r.get("description", ""))
                # "rate" (canonical) or "unit_price" (alternate)
                rate_val = r.get("rate", r.get("unit_price", 0))
                unit = r.get("unit", "ea")
                rates.append(ContractRate(
                    item_description=desc,
                    unit=unit,
                    rate=Decimal(str(rate_val)),
                ))

            # Volume discounts: accept array or single object
            raw_vd = rec.get("volume_discounts", [])
            if not raw_vd:
                # Try singular "volume_discount"
                single_vd = rec.get("volume_discount")
                if single_vd and isinstance(single_vd, dict):
                    raw_vd = [single_vd]
            if not isinstance(raw_vd, list):
                raw_vd = [raw_vd] if raw_vd else []
            volume_discounts = []
            for vd in raw_vd:
                if not isinstance(vd, dict):
                    continue
                thresh = vd.get("threshold_quantity", vd.get("threshold", 0))
                disc = vd.get("discount_pct", 0)
                volume_discounts.append(VolumeDiscount(
                    threshold_quantity=int(thresh),
                    discount_pct=Decimal(str(disc)),
                ))

            # Escalation: accept "annual_escalation_pct" or nested "escalation_clause"
            escalation = rec.get("annual_escalation_pct")
            if escalation is None:
                esc_clause = rec.get("escalation_clause")
                if isinstance(esc_clause, dict):
                    escalation = esc_clause.get("max_annual_increase_pct")

            # Dates: accept "contract_start_date" or "contract_start"
            start_date_str = str(rec.get("contract_start_date", rec.get("contract_start", "")))
            end_date_str = str(rec.get("contract_end_date", rec.get("contract_end", "")))

            # Scope: accept "scope_of_work" or "scope"
            scope = rec.get("scope_of_work", rec.get("scope", []))

            try:
                contracts.append(ContractTerms(
                    vendor_id=str(rec.get("vendor_id", "")),
                    contract_start_date=parse_date(start_date_str),
                    contract_end_date=parse_date(end_date_str),
                    auto_renewal=bool(rec.get("auto_renewal", False)),
                    auto_renewal_notice_days=int(rec.get("auto_renewal_notice_days", 0)),
                    payment_terms=str(rec.get("payment_terms", "")),
                    rates=rates,
                    volume_discounts=volume_discounts,
                    scope_of_work=scope,
                    annual_escalation_pct=(
                        Decimal(str(escalation)) if escalation is not None else None
                    ),
                ))
            except Exception as exc:
                logger.warning(
                    "Skipping contract record for vendor %s: %s",
                    rec.get("vendor_id", "unknown"), exc,
                )

        logger.info("Loaded %d contracts from %s", len(contracts), path.name)
        return contracts

    def load_approval_thresholds(self, filepath: str) -> list[ApprovalThreshold]:
        """Load approval threshold matrix from JSON."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

        records = raw if isinstance(raw, list) else [raw]
        thresholds: list[ApprovalThreshold] = []
        for rec in records:
            max_amt = rec.get("max_amount")
            thresholds.append(ApprovalThreshold(
                level=str(rec["level"]),
                max_amount=Decimal(str(max_amt)) if max_amt is not None else None,
            ))

        logger.info("Loaded %d approval thresholds from %s", len(thresholds), path.name)
        return thresholds

    def load_receipts(self, filepath: str) -> pd.DataFrame:
        """Load goods receipt / delivery records."""
        return self.load(filepath)
