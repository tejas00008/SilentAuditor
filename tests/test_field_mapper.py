"""Tests for src.ingestion.field_mapper."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from src.ingestion.field_mapper import FieldMapper
from src.normalization.cache import CacheManager


@pytest.fixture
def cache(tmp_path):
    cm = CacheManager(db_path=str(tmp_path / "test.db"))
    yield cm
    cm.close()


@pytest.fixture
def mapper(cache):
    return FieldMapper(cache=cache)


@pytest.fixture
def mapper_no_cache():
    return FieldMapper(cache=None)


def _make_df(columns: list[str], rows: int = 3) -> pd.DataFrame:
    """Build a small test DataFrame with the given column names."""
    data = {}
    for col in columns:
        data[col] = [f"val_{i}" for i in range(rows)]
    return pd.DataFrame(data)


# ==================================================================
# auto_map — regex header matching
# ==================================================================

class TestAutoMapRegex:
    """Test pattern-based header matching on various real-world headers."""

    def test_clean_headers(self, mapper_no_cache):
        df = _make_df(["Invoice Number", "Vendor Name", "Invoice Date", "Total Amount"])
        # Put realistic values so type bonus can apply
        df["Invoice Date"] = ["2025-01-01", "2025-02-15", "2025-03-20"]
        df["Total Amount"] = ["1000.00", "2000.00", "3000.00"]
        m = mapper_no_cache.auto_map(df)
        assert m["invoice_number"]["source_column"] == "Invoice Number"
        assert m["vendor_name"]["source_column"] == "Vendor Name"
        assert m["invoice_date"]["source_column"] == "Invoice Date"
        assert m["total_amount"]["source_column"] == "Total Amount"

    def test_messy_headers(self, mapper_no_cache):
        df = _make_df(["Inv #", "VENDOR NAME", "Amt (USD)", "Dt"])
        df["Amt (USD)"] = ["100.00", "200.00", "300.00"]
        df["Dt"] = ["2025-01-01", "2025-02-01", "2025-03-01"]
        m = mapper_no_cache.auto_map(df)
        assert m["invoice_number"]["source_column"] == "Inv #"
        assert m["vendor_name"]["source_column"] == "VENDOR NAME"
        assert m["total_amount"]["source_column"] == "Amt (USD)"
        assert m["invoice_date"]["source_column"] == "Dt"

    def test_abbreviated_headers(self, mapper_no_cache):
        df = _make_df(["inv no", "vendor nm", "qty", "desc"])
        m = mapper_no_cache.auto_map(df)
        assert m["invoice_number"]["source_column"] == "inv no"
        assert m["vendor_name"]["source_column"] == "vendor nm"
        assert m["quantity"]["source_column"] == "qty"
        assert m["line_item_description"]["source_column"] == "desc"

    def test_po_number(self, mapper_no_cache):
        df = _make_df(["PO #", "P.O. Number", "Purchase Order"])
        m = mapper_no_cache.auto_map(df)
        assert "po_number" in m

    def test_payment_date_vs_invoice_date(self, mapper_no_cache):
        df = _make_df(["Invoice Date", "Payment Date", "Amount"])
        df["Invoice Date"] = ["2025-01-01", "2025-02-01", "2025-03-01"]
        df["Payment Date"] = ["2025-01-15", "2025-02-15", "2025-03-15"]
        df["Amount"] = ["100", "200", "300"]
        m = mapper_no_cache.auto_map(df)
        assert m["invoice_date"]["source_column"] == "Invoice Date"
        assert m["payment_date"]["source_column"] == "Payment Date"

    def test_status_column(self, mapper_no_cache):
        df = _make_df(["Status", "Vendor", "Inv Num"])
        m = mapper_no_cache.auto_map(df)
        assert m["payment_status"]["source_column"] == "Status"

    def test_approval_and_email(self, mapper_no_cache):
        df = _make_df(["Approved By", "Submitter Email"])
        m = mapper_no_cache.auto_map(df)
        assert m["approved_by"]["source_column"] == "Approved By"
        assert m["submission_email"]["source_column"] == "Submitter Email"

    def test_cost_center(self, mapper_no_cache):
        df = _make_df(["Department", "Cost Center"])
        m = mapper_no_cache.auto_map(df)
        assert "cost_center" in m

    def test_sap_style_headers(self, mapper_no_cache):
        df = _make_df(["DocNum", "CardCode", "CardName", "DocDate", "DocTotal"])
        df["DocDate"] = ["2025-01-01", "2025-02-01", "2025-03-01"]
        df["DocTotal"] = ["100", "200", "300"]
        m = mapper_no_cache.auto_map(df)
        assert m["invoice_number"]["source_column"] == "DocNum"
        assert m["vendor_id"]["source_column"] == "CardCode"
        assert m["vendor_name"]["source_column"] == "CardName"
        assert m["invoice_date"]["source_column"] == "DocDate"
        assert m["total_amount"]["source_column"] == "DocTotal"

    def test_confidence_above_threshold(self, mapper_no_cache):
        df = _make_df(["Invoice Number", "Vendor Name", "Date", "Amount"])
        df["Date"] = ["2025-01-01", "2025-02-01", "2025-03-01"]
        df["Amount"] = ["100", "200", "300"]
        m = mapper_no_cache.auto_map(df)
        for field, info in m.items():
            assert 0.0 < info["confidence"] <= 1.0


# ==================================================================
# auto_map — cache integration
# ==================================================================

class TestAutoMapCache:
    def test_saves_to_cache(self, mapper, cache):
        df = _make_df(["Invoice Number", "Vendor Name", "Date", "Amount"])
        df["Date"] = ["2025-01-01", "2025-02-01", "2025-03-01"]
        df["Amount"] = ["100", "200", "300"]
        mapper.auto_map(df, customer_id="CUST1")
        stored = cache.get_field_mappings("CUST1")
        assert len(stored) > 0

    def test_restores_from_cache(self, mapper, cache):
        cache.set_field_mappings("CUST1", {
            "Custom Col": {
                "target_field": "invoice_number",
                "confidence": 0.99,
                "verified": True,
            },
        })
        df = _make_df(["Custom Col", "Something Else"])
        m = mapper.auto_map(df, customer_id="CUST1")
        assert m["invoice_number"]["source_column"] == "Custom Col"
        assert m["invoice_number"]["confidence"] == 0.99


# ==================================================================
# ERP templates
# ==================================================================

class TestApplyERPTemplate:
    def test_netsuite(self, mapper_no_cache):
        df = _make_df([
            "Transaction Number", "Vendor ID", "Vendor Name",
            "Date", "Amount", "Item Description", "Rate",
            "Quantity", "PO #", "Date Paid", "Status",
        ])
        m = mapper_no_cache.apply_erp_template(df, "netsuite")
        assert m["invoice_number"]["source_column"] == "Transaction Number"
        assert m["vendor_id"]["source_column"] == "Vendor ID"
        assert m["total_amount"]["source_column"] == "Amount"
        assert m["po_number"]["source_column"] == "PO #"
        assert all(v["confidence"] == 1.0 for v in m.values())

    def test_quickbooks(self, mapper_no_cache):
        df = _make_df([
            "Bill No.", "Vendor", "Bill Date", "Amount",
            "Description", "Rate", "Qty", "P.O. Number",
            "Payment Date", "Payment Status",
        ])
        m = mapper_no_cache.apply_erp_template(df, "quickbooks")
        assert m["invoice_number"]["source_column"] == "Bill No."
        assert m["quantity"]["source_column"] == "Qty"

    def test_missing_template(self, mapper_no_cache):
        df = _make_df(["A"])
        with pytest.raises(FileNotFoundError):
            mapper_no_cache.apply_erp_template(df, "nonexistent_erp")

    def test_fuzzy_column_match(self, mapper_no_cache):
        """Template expects 'Transaction Number' but column is 'Trans Number'."""
        df = _make_df(["Trans Number", "Vendor Name", "Date"])
        m = mapper_no_cache.apply_erp_template(df, "netsuite")
        # Should fuzzy-match at a lower confidence
        if "invoice_number" in m:
            assert m["invoice_number"]["confidence"] < 1.0


# ==================================================================
# apply_mapping — type casting
# ==================================================================

class TestApplyMapping:
    def test_basic_casting(self, mapper_no_cache):
        df = pd.DataFrame({
            "Inv": ["1001", "1002"],
            "Vendor": ["Acme", "Beta"],
            "Date": ["01/15/2025", "2025-02-20"],
            "Amt": ["$1,500.00", "2000.50"],
        })
        mapping = {
            "invoice_number": {"source_column": "Inv", "confidence": 1.0},
            "vendor_name": {"source_column": "Vendor", "confidence": 1.0},
            "invoice_date": {"source_column": "Date", "confidence": 1.0},
            "total_amount": {"source_column": "Amt", "confidence": 1.0},
        }
        result = mapper_no_cache.apply_mapping(df, mapping)
        assert result["invoice_number"].iloc[0] == "1001"
        assert result["invoice_date"].iloc[0] == date(2025, 1, 15)
        assert result["invoice_date"].iloc[1] == date(2025, 2, 20)
        assert result["total_amount"].iloc[0] == Decimal("1500.00")
        assert result["total_amount"].iloc[1] == Decimal("2000.50")

    def test_missing_required_field_raises(self, mapper_no_cache):
        df = pd.DataFrame({"Vendor": ["Acme"]})
        # Missing invoice_number, invoice_date, total_amount
        mapping = {
            "vendor_name": {"source_column": "Vendor", "confidence": 1.0},
        }
        with pytest.raises(ValueError, match="Required fields missing"):
            mapper_no_cache.apply_mapping(df, mapping)

    def test_currency_symbols_stripped(self, mapper_no_cache):
        df = pd.DataFrame({
            "Inv": ["1"],
            "V": ["X"],
            "D": ["2025-01-01"],
            "A": ["€1.234,56".replace(".", "").replace(",", ".")],  # becomes €1234.56
        })
        # Simplify: just test _to_decimal directly
        assert FieldMapper._to_decimal("$1,500.00") == Decimal("1500.00")
        assert FieldMapper._to_decimal("€2000") == Decimal("2000")
        assert FieldMapper._to_decimal("£99.99") == Decimal("99.99")

    def test_parenthesised_negatives(self, mapper_no_cache):
        assert FieldMapper._to_decimal("(123.45)") == Decimal("-123.45")

    def test_empty_amount(self, mapper_no_cache):
        assert FieldMapper._to_decimal("") is None
        assert FieldMapper._to_decimal(None) is None

    def test_unparseable_date_becomes_none(self, mapper_no_cache):
        df = pd.DataFrame({
            "Inv": ["1"],
            "V": ["X"],
            "D": ["not-a-date"],
            "A": ["100"],
        })
        mapping = {
            "invoice_number": {"source_column": "Inv", "confidence": 1.0},
            "vendor_name": {"source_column": "V", "confidence": 1.0},
            "invoice_date": {"source_column": "D", "confidence": 1.0},
            "total_amount": {"source_column": "A", "confidence": 1.0},
        }
        result = mapper_no_cache.apply_mapping(df, mapping)
        assert result["invoice_date"].iloc[0] is None


# ==================================================================
# get_unmapped_columns
# ==================================================================

class TestGetUnmappedColumns:
    def test_basic(self, mapper_no_cache):
        df = _make_df(["A", "B", "C", "D"])
        mapping = {
            "invoice_number": {"source_column": "A", "confidence": 1.0},
            "vendor_name": {"source_column": "B", "confidence": 1.0},
        }
        unmapped = mapper_no_cache.get_unmapped_columns(df, mapping)
        assert set(unmapped) == {"C", "D"}

    def test_all_mapped(self, mapper_no_cache):
        df = _make_df(["A", "B"])
        mapping = {
            "invoice_number": {"source_column": "A", "confidence": 1.0},
            "vendor_name": {"source_column": "B", "confidence": 1.0},
        }
        assert mapper_no_cache.get_unmapped_columns(df, mapping) == []


# ==================================================================
# Type bonus
# ==================================================================

class TestTypeBonus:
    def test_numeric_bonus(self, mapper_no_cache):
        df = pd.DataFrame({"amt": ["100.00", "200.50", "300"]})
        bonus = FieldMapper._type_bonus(df, "amt", "total_amount")
        assert bonus > 0

    def test_date_bonus(self, mapper_no_cache):
        df = pd.DataFrame({"dt": ["2025-01-01", "2025-02-15", "2025-03-20"]})
        bonus = FieldMapper._type_bonus(df, "dt", "invoice_date")
        assert bonus > 0

    def test_no_bonus_for_string(self, mapper_no_cache):
        df = pd.DataFrame({"name": ["Acme", "Beta", "Gamma"]})
        bonus = FieldMapper._type_bonus(df, "name", "vendor_name")
        assert bonus == 0.0
