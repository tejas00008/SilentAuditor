"""Auto-detect and map customer column names to the SilentAuditor schema."""

import logging
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.normalization.cache import CacheManager
from src.utils.date_utils import parse_date
from src.utils.similarity import fuzzy_match_score

logger = logging.getLogger(__name__)

# Path to bundled ERP templates.
_ERP_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "erp_templates"


# ------------------------------------------------------------------
# Header-matching patterns
# ------------------------------------------------------------------
# Each entry maps a compiled regex (tested against the lowered, stripped
# column header) to a schema field name.  Patterns are evaluated in order;
# the first match wins.  More specific patterns come first.

# Separator: matches zero or more whitespace, underscores, or hyphens
# between words in column headers (e.g. "vendor_name", "vendor name",
# "vendor-name").
_S = r"[\s_\-]*"

_HEADER_PATTERNS: list[tuple[re.Pattern, str]] = [
    # invoice_number
    (re.compile(rf"inv(?:oice)?{_S}(?:num|no|#|id|nbr|number)", re.I), "invoice_number"),
    (re.compile(rf"bill{_S}(?:num|no|#|id|nbr|number)", re.I), "invoice_number"),
    (re.compile(rf"transaction{_S}(?:num|no|#|number)", re.I), "invoice_number"),
    (re.compile(rf"doc{_S}num", re.I), "invoice_number"),

    # vendor_id
    (re.compile(rf"(?:vendor|supplier|vend){_S}(?:id|code|no|num|#)", re.I), "vendor_id"),
    (re.compile(rf"card{_S}code", re.I), "vendor_id"),
    (re.compile(rf"supplier{_S}account", re.I), "vendor_id"),
    (re.compile(rf"contact{_S}id", re.I), "vendor_id"),

    # vendor_name  (must come after vendor_id so "vendor id" doesn't match here)
    (re.compile(rf"(?:vendor|supplier|vend){_S}(?:name|nm)", re.I), "vendor_name"),
    (re.compile(rf"card{_S}name", re.I), "vendor_name"),
    (re.compile(rf"supplier{_S}name", re.I), "vendor_name"),
    (re.compile(rf"contact{_S}name", re.I), "vendor_name"),

    # payment_date  (before generic "date" patterns)
    (re.compile(rf"(?:pay|payment|paid|remit){_S}(?:date|dt)", re.I), "payment_date"),
    (re.compile(rf"date{_S}paid", re.I), "payment_date"),
    (re.compile(rf"fully{_S}paid", re.I), "payment_date"),

    # invoice_date
    (re.compile(rf"inv(?:oice)?{_S}(?:date|dt)", re.I), "invoice_date"),
    (re.compile(rf"bill{_S}date", re.I), "invoice_date"),
    (re.compile(rf"doc{_S}date", re.I), "invoice_date"),
    (re.compile(r"^date$", re.I), "invoice_date"),
    (re.compile(r"^dt$", re.I), "invoice_date"),

    # total_amount
    (re.compile(rf"(?:total|net){_S}(?:amount|amt|sum)", re.I), "total_amount"),
    (re.compile(rf"(?:amount|amt)(?:{_S}\(?\s*usd\s*\)?)?\s*$", re.I), "total_amount"),
    (re.compile(r"^amount$", re.I), "total_amount"),
    (re.compile(r"^amt$", re.I), "total_amount"),
    (re.compile(rf"doc{_S}total", re.I), "total_amount"),

    # unit_price
    (re.compile(rf"unit{_S}(?:price|cost|amt|amount)", re.I), "unit_price"),
    (re.compile(r"^(?:price|rate)$", re.I), "unit_price"),
    (re.compile(rf"unit{_S}amount", re.I), "unit_price"),

    # quantity
    (re.compile(r"^(?:qty|quantity|units|count)$", re.I), "quantity"),

    # line_item_description
    (re.compile(rf"(?:line{_S})?item{_S}desc", re.I), "line_item_description"),
    (re.compile(r"^desc(?:ription)?$", re.I), "line_item_description"),
    (re.compile(r"^detail(?:s)?$", re.I), "line_item_description"),
    (re.compile(r"^memo$", re.I), "line_item_description"),
    (re.compile(r"dscription", re.I), "line_item_description"),  # SAP typo

    # po_number
    (re.compile(rf"p\.?{_S}o\.?{_S}(?:num|no|#|number)?", re.I), "po_number"),
    (re.compile(rf"purchase{_S}order", re.I), "po_number"),
    (re.compile(rf"num{_S}at{_S}card", re.I), "po_number"),

    # payment_status
    (re.compile(rf"pay(?:ment)?{_S}status", re.I), "payment_status"),
    (re.compile(r"^status$", re.I), "payment_status"),
    (re.compile(rf"doc{_S}status", re.I), "payment_status"),

    # project_code
    (re.compile(rf"project(?:{_S}code)?", re.I), "project_code"),
    (re.compile(r"^job$", re.I), "project_code"),

    # cost_center
    (re.compile(rf"cost{_S}cent(?:er|re)", re.I), "cost_center"),
    (re.compile(r"^dept(?:artment)?$", re.I), "cost_center"),
    (re.compile(r"^department$", re.I), "cost_center"),

    # approved_by
    (re.compile(r"approv", re.I), "approved_by"),

    # submission_email
    (re.compile(r"email", re.I), "submission_email"),
]

# Canonical aliases used for fuzzy matching as a fallback.
_FUZZY_ALIASES: dict[str, list[str]] = {
    "invoice_number": ["invoice number", "invoice no", "inv num", "bill number"],
    "vendor_name": ["vendor name", "supplier name"],
    "vendor_id": ["vendor id", "supplier id", "vendor code"],
    "invoice_date": ["invoice date", "bill date", "date"],
    "total_amount": ["total amount", "amount", "net amount", "invoice amount"],
    "line_item_description": ["description", "item description", "line item"],
    "unit_price": ["unit price", "price", "rate"],
    "quantity": ["quantity", "qty"],
    "po_number": ["po number", "purchase order"],
    "payment_date": ["payment date", "date paid"],
    "payment_status": ["payment status", "status"],
    "project_code": ["project code", "project", "job"],
    "cost_center": ["cost center", "department"],
    "approved_by": ["approved by", "approver"],
    "submission_email": ["email", "submitter email"],
}


class FieldMapper:
    """Auto-detect and map customer column names to the SilentAuditor schema."""

    SCHEMA_FIELDS: dict[str, dict] = {
        "invoice_number":        {"required": True,  "type": "string"},
        "vendor_id":             {"required": False, "type": "string"},
        "vendor_name":           {"required": True,  "type": "string"},
        "invoice_date":          {"required": True,  "type": "date"},
        "total_amount":          {"required": True,  "type": "decimal"},
        "line_item_description": {"required": False, "type": "string"},
        "unit_price":            {"required": False, "type": "decimal"},
        "quantity":              {"required": False, "type": "decimal"},
        "po_number":             {"required": False, "type": "string"},
        "payment_date":          {"required": False, "type": "date"},
        "payment_status":        {"required": False, "type": "string"},
        "project_code":          {"required": False, "type": "string"},
        "cost_center":           {"required": False, "type": "string"},
        "approved_by":           {"required": False, "type": "string"},
        "submission_email":      {"required": False, "type": "string"},
    }

    def __init__(self, cache: Optional[CacheManager] = None) -> None:
        self.cache = cache

    # ------------------------------------------------------------------
    # Auto-mapping
    # ------------------------------------------------------------------

    def auto_map(
        self, df: pd.DataFrame, customer_id: Optional[str] = None,
    ) -> dict:
        """Auto-detect column mappings from *df* to the schema.

        Returns::

            {schema_field: {"source_column": str, "confidence": float}}

        Steps:
        1. Check cached mappings for *customer_id* (if available).
        2. Regex pattern match on column headers.
        3. Fuzzy-match fallback for unresolved columns.
        4. Data-type and value-pattern bonuses.
        """
        # ---- Step 1: cached mappings -----------------------------------
        cached: dict = {}
        if self.cache and customer_id:
            raw = self.cache.get_field_mappings(customer_id)
            for src_col, info in raw.items():
                if src_col in df.columns:
                    target = info["target_field"]
                    cached[target] = {
                        "source_column": src_col,
                        "confidence": info["confidence"],
                    }
            if cached:
                logger.info(
                    "Restored %d cached mappings for customer %s",
                    len(cached), customer_id,
                )

        # ---- Step 2 + 3: header analysis --------------------------------
        mapping: dict = dict(cached)  # start with cached
        used_columns: set[str] = {v["source_column"] for v in mapping.values()}

        for col in df.columns:
            if col in used_columns:
                continue
            header = col.strip()

            # Regex patterns
            match_field = self._regex_match(header)
            if match_field and match_field not in mapping:
                confidence = 0.90
                confidence += self._type_bonus(df, col, match_field)
                mapping[match_field] = {
                    "source_column": col,
                    "confidence": min(confidence, 1.0),
                }
                used_columns.add(col)
                continue

            # Fuzzy fallback
            best_field, best_score = self._fuzzy_match(header)
            if best_field and best_field not in mapping and best_score >= 0.65:
                confidence = round(best_score * 0.8, 2)  # discount fuzzy
                confidence += self._type_bonus(df, col, best_field)
                mapping[best_field] = {
                    "source_column": col,
                    "confidence": min(confidence, 1.0),
                }
                used_columns.add(col)

        # ---- persist to cache ------------------------------------------
        if self.cache and customer_id and mapping:
            cache_payload = {
                v["source_column"]: {
                    "target_field": field,
                    "confidence": v["confidence"],
                    "verified": False,
                }
                for field, v in mapping.items()
            }
            self.cache.set_field_mappings(customer_id, cache_payload)

        return mapping

    # ------------------------------------------------------------------
    # ERP template
    # ------------------------------------------------------------------

    def apply_erp_template(
        self, df: pd.DataFrame, erp_name: str,
    ) -> dict:
        """Apply a pre-built ERP column mapping template.

        Loads ``config/erp_templates/{erp_name}.yaml`` and matches
        template column names against the actual DataFrame columns.
        """
        template_path = _ERP_TEMPLATES_DIR / f"{erp_name}.yaml"
        if not template_path.exists():
            raise FileNotFoundError(
                f"ERP template not found: {template_path}"
            )

        with open(template_path, encoding="utf-8") as fh:
            template = yaml.safe_load(fh)

        field_mapping = template.get("field_mapping", {})
        df_columns_lower = {c.lower().strip(): c for c in df.columns}

        mapping: dict = {}
        for schema_field, erp_column in field_mapping.items():
            erp_lower = erp_column.lower().strip()
            if erp_lower in df_columns_lower:
                mapping[schema_field] = {
                    "source_column": df_columns_lower[erp_lower],
                    "confidence": 1.0,
                }
            else:
                # Try fuzzy matching the template column to actual columns
                best_col: Optional[str] = None
                best_score: float = 0.0
                for actual in df.columns:
                    score = fuzzy_match_score(erp_column, actual)
                    if score > best_score:
                        best_score = score
                        best_col = actual
                if best_col and best_score >= 0.75:
                    mapping[schema_field] = {
                        "source_column": best_col,
                        "confidence": round(best_score, 2),
                    }

        return mapping

    # ------------------------------------------------------------------
    # Apply mapping & type casting
    # ------------------------------------------------------------------

    def apply_mapping(self, df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        """Rename columns per *mapping* and cast to expected types.

        Raises ``ValueError`` if any required field is missing from the
        mapping.
        """
        # Validate required fields
        missing = [
            field for field, meta in self.SCHEMA_FIELDS.items()
            if meta["required"] and field not in mapping
        ]
        if missing:
            raise ValueError(
                f"Required fields missing from mapping: {missing}"
            )

        # Build rename dict: source_column → schema_field
        rename_map = {
            v["source_column"]: field
            for field, v in mapping.items()
        }
        mapped_cols = list(rename_map.keys())
        result = df[mapped_cols].copy()
        result.rename(columns=rename_map, inplace=True)

        # Type casting
        for field, meta in self.SCHEMA_FIELDS.items():
            if field not in result.columns:
                continue
            if meta["type"] == "date":
                result[field] = result[field].apply(
                    lambda v: parse_date(str(v)) if pd.notna(v) and str(v).strip() else None
                )
            elif meta["type"] == "decimal":
                result[field] = result[field].apply(self._to_decimal)

        return result

    # ------------------------------------------------------------------
    # Unmapped columns
    # ------------------------------------------------------------------

    def get_unmapped_columns(
        self, df: pd.DataFrame, mapping: dict,
    ) -> list[str]:
        """Return DataFrame columns that were not mapped to any field."""
        mapped_sources = {v["source_column"] for v in mapping.values()}
        return [c for c in df.columns if c not in mapped_sources]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _regex_match(header: str) -> Optional[str]:
        """Return the schema field matched by regex, or ``None``."""
        for pattern, field in _HEADER_PATTERNS:
            if pattern.search(header):
                return field
        return None

    @staticmethod
    def _fuzzy_match(header: str) -> tuple[Optional[str], float]:
        """Fuzzy-match *header* against known aliases.

        Returns ``(field_name, score)`` for the best match.
        """
        best_field: Optional[str] = None
        best_score: float = 0.0
        for field, aliases in _FUZZY_ALIASES.items():
            for alias in aliases:
                score = fuzzy_match_score(header.lower(), alias)
                if score > best_score:
                    best_score = score
                    best_field = field
        return best_field, best_score

    @staticmethod
    def _type_bonus(df: pd.DataFrame, col: str, field: str) -> float:
        """Return a small confidence bonus if data types match expectations."""
        meta = FieldMapper.SCHEMA_FIELDS.get(field)
        if meta is None:
            return 0.0

        sample = df[col].dropna().head(20)
        if sample.empty:
            return 0.0

        if meta["type"] == "decimal":
            parseable = sum(1 for v in sample if _is_numeric(str(v)))
            if parseable / len(sample) >= 0.8:
                return 0.10
        elif meta["type"] == "date":
            parseable = sum(1 for v in sample if parse_date(str(v)) is not None)
            if parseable / len(sample) >= 0.6:
                return 0.10
        return 0.0

    @staticmethod
    def _to_decimal(value: object) -> Optional[Decimal]:
        """Convert a value to Decimal, handling currency symbols and commas."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        s = str(value).strip()
        if not s:
            return None
        # Strip currency symbols and thousands separators
        s = s.replace("$", "").replace("€", "").replace("£", "")
        s = s.replace(",", "")
        # Handle parenthesised negatives: (123.45) → -123.45
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        try:
            return Decimal(s)
        except InvalidOperation:
            return None


def _is_numeric(s: str) -> bool:
    """Check whether *s* looks like a number (with optional currency / commas)."""
    cleaned = s.strip().replace("$", "").replace("€", "").replace("£", "")
    cleaned = cleaned.replace(",", "")
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    try:
        Decimal(cleaned)
        return True
    except InvalidOperation:
        return False
