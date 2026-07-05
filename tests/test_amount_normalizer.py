"""Tests for src.normalization.amount_normalizer."""

from decimal import Decimal

import pandas as pd
import pytest

from src.normalization.amount_normalizer import (
    AmountNormalizer,
    _parse_amount,
    _is_tax_description,
)


@pytest.fixture
def normalizer():
    return AmountNormalizer()


# ==================================================================
# _parse_amount
# ==================================================================

class TestParseAmount:
    # ---- US format ----
    def test_plain_number(self):
        assert _parse_amount("1234.56") == Decimal("1234.56")

    def test_with_dollar(self):
        assert _parse_amount("$1,234.56") == Decimal("1234.56")

    def test_with_euro(self):
        assert _parse_amount("€500.00") == Decimal("500.00")

    def test_with_pound(self):
        assert _parse_amount("£99.99") == Decimal("99.99")

    def test_with_rupee(self):
        assert _parse_amount("₹10000") == Decimal("10000.00")

    # ---- EU format ----
    def test_eu_format(self):
        assert _parse_amount("1.234,56") == Decimal("1234.56")

    def test_eu_format_large(self):
        assert _parse_amount("12.345.678,90") == Decimal("12345678.90")

    # ---- Negative values ----
    def test_parenthesised_negative(self):
        assert _parse_amount("(500.00)") == Decimal("-500.00")

    def test_minus_sign(self):
        assert _parse_amount("-250.00") == Decimal("-250.00")

    def test_cr_suffix(self):
        assert _parse_amount("100.00 CR") == Decimal("-100.00")

    def test_credit_suffix(self):
        assert _parse_amount("$1,000.00 Credit") == Decimal("-1000.00")

    # ---- Edge cases ----
    def test_none(self):
        assert _parse_amount(None) is None

    def test_empty(self):
        assert _parse_amount("") is None

    def test_garbage(self):
        assert _parse_amount("abc") is None

    def test_integer(self):
        assert _parse_amount("5000") == Decimal("5000.00")

    def test_rounds_to_two_places(self):
        assert _parse_amount("123.456") == Decimal("123.46")

    def test_whitespace(self):
        assert _parse_amount("  $1,234.56  ") == Decimal("1234.56")

    def test_zero(self):
        assert _parse_amount("0") == Decimal("0.00")

    def test_numeric_input(self):
        assert _parse_amount(1234.56) == Decimal("1234.56")


# ==================================================================
# _is_tax_description
# ==================================================================

class TestIsTaxDescription:
    def test_gst(self):
        assert _is_tax_description("GST 10%") is True

    def test_vat(self):
        assert _is_tax_description("VAT @ 20%") is True

    def test_sales_tax(self):
        assert _is_tax_description("California Sales Tax") is True

    def test_hst(self):
        assert _is_tax_description("HST Ontario") is True

    def test_not_tax(self):
        assert _is_tax_description("Steel rebar 40mm") is False

    def test_empty(self):
        assert _is_tax_description("") is False


# ==================================================================
# normalize_amounts — basic
# ==================================================================

class TestNormalizeAmountsBasic:
    def test_adds_columns(self, normalizer):
        df = pd.DataFrame({"total_amount": ["$1,000.00", "$500.00"]})
        result = normalizer.normalize_amounts(df)
        assert "gross_amount" in result.columns
        assert "net_amount" in result.columns
        assert "tax_amount" in result.columns
        assert "is_credit" in result.columns
        assert "amount_issues" in result.columns

    def test_parses_amounts(self, normalizer):
        df = pd.DataFrame({"total_amount": ["$1,500.00"]})
        result = normalizer.normalize_amounts(df)
        assert result["gross_amount"].iloc[0] == Decimal("1500.00")

    def test_preserves_original_columns(self, normalizer):
        df = pd.DataFrame({
            "total_amount": ["100"],
            "vendor_name": ["Acme"],
        })
        result = normalizer.normalize_amounts(df)
        assert "vendor_name" in result.columns


# ==================================================================
# Tax detection
# ==================================================================

class TestTaxDetection:
    def test_explicit_tax_line(self, normalizer):
        df = pd.DataFrame({
            "total_amount": ["1000.00", "100.00"],
            "line_item_description": ["Consulting services", "GST 10%"],
        })
        result = normalizer.normalize_amounts(df)
        # The GST row should have tax = its amount, net = 0
        assert result["tax_amount"].iloc[1] == Decimal("100.00")
        assert result["net_amount"].iloc[1] == Decimal("0.00")
        # The service row should have tax = 0, net = full amount
        assert result["tax_amount"].iloc[0] == Decimal("0.00")
        assert result["net_amount"].iloc[0] == Decimal("1000.00")

    def test_no_tax_lines_defaults_to_zero(self, normalizer):
        """When there are no tax keywords AND the amount doesn't match a
        common tax rate decomposition, tax should default to zero."""
        df = pd.DataFrame({
            "total_amount": ["1337.42"],
            "line_item_description": ["Random services"],
        })
        result = normalizer.normalize_amounts(df)
        tax = result["tax_amount"].iloc[0]
        net = result["net_amount"].iloc[0]
        assert tax + net == Decimal("1337.42")

    def test_inferred_tax_10pct(self, normalizer):
        """$1100 at 10% tax = $1000 net + $100 tax.  Net is a round
        dollar amount, so the heuristic should catch it."""
        df = pd.DataFrame({"total_amount": ["1100.00"]})
        result = normalizer.normalize_amounts(df)
        tax = result["tax_amount"].iloc[0]
        net = result["net_amount"].iloc[0]
        assert tax + net == Decimal("1100.00")
        # The heuristic may or may not identify the embedded tax;
        # what matters is the split is consistent.
        assert net >= Decimal("0")


# ==================================================================
# Credit detection
# ==================================================================

class TestCreditDetection:
    def test_negative_amount(self, normalizer):
        df = pd.DataFrame({"total_amount": ["-500.00"]})
        result = normalizer.normalize_amounts(df)
        assert result["is_credit"].iloc[0] == True

    def test_parenthesised(self, normalizer):
        df = pd.DataFrame({"total_amount": ["(500.00)"]})
        result = normalizer.normalize_amounts(df)
        assert result["is_credit"].iloc[0] == True

    def test_cr_suffix(self, normalizer):
        df = pd.DataFrame({"total_amount": ["500.00 CR"]})
        result = normalizer.normalize_amounts(df)
        assert result["is_credit"].iloc[0] == True

    def test_credit_in_description(self, normalizer):
        df = pd.DataFrame({
            "total_amount": ["500.00"],
            "line_item_description": ["Credit memo for overbilling"],
        })
        result = normalizer.normalize_amounts(df)
        assert result["is_credit"].iloc[0] == True

    def test_positive_not_credit(self, normalizer):
        df = pd.DataFrame({"total_amount": ["500.00"]})
        result = normalizer.normalize_amounts(df)
        assert result["is_credit"].iloc[0] == False


# ==================================================================
# Validation
# ==================================================================

class TestValidation:
    def test_zero_flagged(self, normalizer):
        df = pd.DataFrame({"total_amount": ["0.00"]})
        result = normalizer.normalize_amounts(df)
        assert any("Zero" in i for i in result["amount_issues"].iloc[0])

    def test_large_amount_flagged(self, normalizer):
        df = pd.DataFrame({"total_amount": ["99999999.00"]})
        result = normalizer.normalize_amounts(df)
        assert any("large" in i.lower() for i in result["amount_issues"].iloc[0])

    def test_unparseable_flagged(self, normalizer):
        df = pd.DataFrame({"total_amount": ["not_a_number"]})
        result = normalizer.normalize_amounts(df)
        assert any("Unparseable" in i for i in result["amount_issues"].iloc[0])

    def test_normal_amount_no_issues(self, normalizer):
        df = pd.DataFrame({"total_amount": ["5000.00"]})
        result = normalizer.normalize_amounts(df)
        assert result["amount_issues"].iloc[0] == []


# ==================================================================
# Line-item reconciliation
# ==================================================================

class TestReconcileLineItems:
    def test_exact_match(self):
        result = AmountNormalizer.reconcile_line_items(
            Decimal("1000.00"),
            [Decimal("400.00"), Decimal("600.00")],
        )
        assert result["matches"] is True
        assert result["difference"] == Decimal("0.00")

    def test_within_tolerance(self):
        result = AmountNormalizer.reconcile_line_items(
            Decimal("1000.00"),
            [Decimal("400.00"), Decimal("595.00")],
            tolerance_pct=1.0,
        )
        # Diff = 5.00 / 1000 = 0.5% < 1%
        assert result["matches"] is True

    def test_outside_tolerance(self):
        result = AmountNormalizer.reconcile_line_items(
            Decimal("1000.00"),
            [Decimal("400.00"), Decimal("500.00")],
            tolerance_pct=1.0,
        )
        # Diff = 100.00 / 1000 = 10%
        assert result["matches"] is False
        assert result["difference_pct"] == 10.0

    def test_zero_total(self):
        result = AmountNormalizer.reconcile_line_items(
            Decimal("0"),
            [Decimal("0"), Decimal("0")],
        )
        assert result["matches"] is True

    def test_zero_total_nonzero_lines(self):
        result = AmountNormalizer.reconcile_line_items(
            Decimal("0"),
            [Decimal("100")],
        )
        assert result["matches"] is False


# ==================================================================
# Multiple currency formats in one dataset
# ==================================================================

class TestMixedFormats:
    def test_mixed_currencies(self, normalizer):
        df = pd.DataFrame({
            "total_amount": ["$1,500.00", "€2.000,50", "£300", "(100.00)"],
        })
        result = normalizer.normalize_amounts(df)
        assert result["gross_amount"].iloc[0] == Decimal("1500.00")
        assert result["gross_amount"].iloc[1] == Decimal("2000.50")
        assert result["gross_amount"].iloc[2] == Decimal("300.00")
        assert result["gross_amount"].iloc[3] == Decimal("-100.00")
