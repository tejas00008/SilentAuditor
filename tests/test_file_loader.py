"""Tests for src.ingestion.file_loader."""

import json
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from src.ingestion.file_loader import FileLoader
from src.utils.constants import ApprovalThreshold, ContractTerms


@pytest.fixture
def loader():
    return FileLoader()


# ------------------------------------------------------------------
# CSV loading
# ------------------------------------------------------------------

class TestLoadCSV:
    def test_basic_csv(self, loader, tmp_path):
        f = tmp_path / "invoices.csv"
        f.write_text("inv_no,vendor,amount\n1001,Acme,500.00\n1002,Beta,750.50\n")
        df = loader.load(str(f))
        assert len(df) == 2
        assert list(df.columns) == ["inv_no", "vendor", "amount"]

    def test_semicolon_delimiter(self, loader, tmp_path):
        f = tmp_path / "semi.csv"
        f.write_text("a;b;c\n1;2;3\n")
        df = loader.load(str(f))
        assert len(df.columns) == 3

    def test_tab_delimiter(self, loader, tmp_path):
        f = tmp_path / "data.tsv"
        f.write_text("col1\tcol2\n1\t2\n")
        df = loader.load(str(f))
        assert len(df.columns) == 2

    def test_whitespace_stripped(self, loader, tmp_path):
        f = tmp_path / "ws.csv"
        f.write_text("  name  ,  amount \n  Acme  ,  100  \n")
        df = loader.load(str(f))
        assert df.columns[0] == "name"
        assert df.iloc[0]["name"] == "Acme"

    def test_latin1_encoding(self, loader, tmp_path):
        f = tmp_path / "latin.csv"
        f.write_bytes("name,city\nCafé,Zürich\n".encode("latin-1"))
        df = loader.load(str(f))
        assert "Café" in df["name"].values

    def test_file_not_found(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.csv")


# ------------------------------------------------------------------
# Excel loading
# ------------------------------------------------------------------

class TestLoadExcel:
    def test_basic_xlsx(self, loader, tmp_path):
        f = tmp_path / "invoices.xlsx"
        pd.DataFrame({"inv": ["1001"], "amt": ["500"]}).to_excel(
            f, index=False, engine="openpyxl"
        )
        df = loader.load(str(f))
        assert len(df) == 1
        assert "inv" in df.columns


# ------------------------------------------------------------------
# JSON loading
# ------------------------------------------------------------------

class TestLoadJSON:
    def test_array_of_objects(self, loader, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps([{"a": 1, "b": 2}, {"a": 3, "b": 4}]))
        df = loader.load(str(f))
        assert len(df) == 2

    def test_nested_structure(self, loader, tmp_path):
        f = tmp_path / "nested.json"
        f.write_text(json.dumps({
            "meta": {"version": 1},
            "invoices": [{"id": "1"}, {"id": "2"}],
        }))
        df = loader.load(str(f))
        assert len(df) == 2
        assert "id" in df.columns

    def test_single_object(self, loader, tmp_path):
        f = tmp_path / "single.json"
        f.write_text(json.dumps({"a": 1, "b": 2}))
        df = loader.load(str(f))
        assert len(df) == 1


# ------------------------------------------------------------------
# Unsupported extension
# ------------------------------------------------------------------

class TestUnsupportedFormat:
    def test_raises(self, loader, tmp_path):
        f = tmp_path / "data.xml"
        f.write_text("<root/>")
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(str(f))


# ------------------------------------------------------------------
# Contract loading
# ------------------------------------------------------------------

class TestLoadContracts:
    def test_basic(self, loader, tmp_path):
        f = tmp_path / "contracts.json"
        f.write_text(json.dumps([{
            "vendor_id": "V001",
            "contract_start_date": "2025-01-01",
            "contract_end_date": "2025-12-31",
            "auto_renewal": True,
            "auto_renewal_notice_days": 30,
            "payment_terms": "Net 30",
            "rates": [{"item_description": "Consulting", "unit": "hour", "rate": "150.00"}],
            "volume_discounts": [{"threshold_quantity": 100, "discount_pct": "5.0"}],
            "scope_of_work": ["IT consulting", "System integration"],
            "annual_escalation_pct": "3.0",
        }]))
        contracts = loader.load_contracts(str(f))
        assert len(contracts) == 1
        c = contracts[0]
        assert isinstance(c, ContractTerms)
        assert c.vendor_id == "V001"
        assert c.contract_start_date == date(2025, 1, 1)
        assert c.rates[0].rate == Decimal("150.00")
        assert c.volume_discounts[0].discount_pct == Decimal("5.0")
        assert c.annual_escalation_pct == Decimal("3.0")


# ------------------------------------------------------------------
# Approval thresholds
# ------------------------------------------------------------------

class TestLoadApprovalThresholds:
    def test_basic(self, loader, tmp_path):
        f = tmp_path / "thresholds.json"
        f.write_text(json.dumps([
            {"level": "Manager", "max_amount": "5000"},
            {"level": "Director", "max_amount": "25000"},
            {"level": "VP", "max_amount": None},
        ]))
        thresholds = loader.load_approval_thresholds(str(f))
        assert len(thresholds) == 3
        assert thresholds[0].level == "Manager"
        assert thresholds[0].max_amount == Decimal("5000")
        assert thresholds[2].max_amount is None
