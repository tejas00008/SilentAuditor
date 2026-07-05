"""Tests for src.ingestion.data_quality.DataQualityAssessor."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from src.ingestion.data_quality import DataQualityAssessor
from src.utils.constants import ContractTerms, ContractRate, DataQualityReport


@pytest.fixture
def assessor():
    return DataQualityAssessor()


# ------------------------------------------------------------------
# Helper factories
# ------------------------------------------------------------------

def _full_df(n: int = 20, months: int = 12) -> pd.DataFrame:
    """DataFrame with every expected field populated, spanning *months*."""
    rows = []
    for i in range(n):
        month = (i % months) + 1
        rows.append({
            "invoice_number": f"INV-{i:04d}",
            "vendor_id": f"V{(i % 5):03d}",
            "vendor_name": f"Vendor {i % 5}",
            "invoice_date": date(2024, month, 15),
            "total_amount": Decimal(str(1000 + i * 100)),
            "line_item_description": f"Service item {i}",
            "unit_price": Decimal(str(50 + i)),
            "quantity": Decimal("10"),
            "po_number": f"PO-{i:04d}",
            "payment_date": date(2024, month, 28),
            "approved_by": f"Manager {i % 3}",
        })
    return pd.DataFrame(rows)


def _minimal_df(n: int = 10, months: int = 12) -> pd.DataFrame:
    """Only the four required fields, spanning *months* calendar months.

    Ensures the first row is in month 1 and the last row is in
    month ``months`` so the date range covers exactly ``months - 1``
    calendar months (e.g. months=7 → Jan–Jul → 6 months span).
    """
    rows = []
    for i in range(max(n, months)):
        month = (i % months) + 1
        rows.append({
            "invoice_number": f"INV-{i:04d}",
            "vendor_name": f"Vendor {i % 3}",
            "invoice_date": date(2024, month, 15),
            "total_amount": Decimal(str(500 + i * 50)),
        })
    return pd.DataFrame(rows)


def _make_contract() -> ContractTerms:
    return ContractTerms(
        vendor_id="V001",
        contract_start_date=date(2024, 1, 1),
        contract_end_date=date(2024, 12, 31),
        auto_renewal=False,
        auto_renewal_notice_days=30,
        payment_terms="Net 30",
        rates=[ContractRate(item_description="Consulting", unit="hour", rate=Decimal("150"))],
        volume_discounts=[],
        scope_of_work=["IT consulting"],
    )


# ==================================================================
# Complete data
# ==================================================================

class TestFullData:
    def test_returns_report(self, assessor):
        report = assessor.assess(
            _full_df(),
            vendor_df=pd.DataFrame({"vendor_id": ["V001"], "name": ["Acme"]}),
            contracts=[_make_contract()],
            receipts_df=pd.DataFrame({"receipt_id": ["R1"]}),
            approval_thresholds=[{"level": "Manager", "max_amount": 5000}],
        )
        assert isinstance(report, DataQualityReport)

    def test_all_modules_can_run(self, assessor):
        report = assessor.assess(
            _full_df(),
            vendor_df=pd.DataFrame({"vendor_id": ["V001"]}),
            contracts=[_make_contract()],
            receipts_df=pd.DataFrame({"receipt_id": ["R1"]}),
            approval_thresholds=[{"level": "Manager", "max_amount": 5000}],
        )
        for module, info in report.module_readiness.items():
            assert info["can_run"] is True, f"{module} should be able to run"

    def test_high_overall_score(self, assessor):
        report = assessor.assess(
            _full_df(),
            vendor_df=pd.DataFrame({"vendor_id": ["V001"]}),
            contracts=[_make_contract()],
            receipts_df=pd.DataFrame({"receipt_id": ["R1"]}),
            approval_thresholds=[{"level": "Manager", "max_amount": 5000}],
        )
        assert report.overall_readiness_score >= 0.70

    def test_statistics(self, assessor):
        df = _full_df(n=20)
        report = assessor.assess(df)
        assert report.total_invoices == 20
        assert report.unique_vendors == 5
        assert report.date_range["earliest"] is not None
        assert report.date_range["latest"] is not None

    def test_field_completeness_all_present(self, assessor):
        report = assessor.assess(_full_df())
        for field in ("invoice_number", "vendor_name", "total_amount"):
            assert report.fields_present[field]["present"] is True
            assert report.fields_present[field]["completeness"] == 1.0

    def test_effectiveness_with_line_items(self, assessor):
        report = assessor.assess(_full_df())
        dup = report.module_readiness["duplicate_detection"]
        assert dup["effectiveness"] >= 0.95


# ==================================================================
# Minimal data (required fields only)
# ==================================================================

class TestMinimalData:
    def test_core_modules_run(self, assessor):
        report = assessor.assess(_minimal_df())
        assert report.module_readiness["duplicate_detection"]["can_run"] is True
        assert report.module_readiness["split_invoicing"]["can_run"] is True

    def test_price_creep_blocked(self, assessor):
        report = assessor.assess(_minimal_df())
        assert report.module_readiness["price_creep"]["can_run"] is False

    def test_market_price_blocked(self, assessor):
        report = assessor.assess(_minimal_df())
        assert report.module_readiness["market_price"]["can_run"] is False

    def test_phantom_blocked_without_descriptions(self, assessor):
        report = assessor.assess(_minimal_df())
        assert report.module_readiness["phantom_services"]["can_run"] is False

    def test_contract_blocked(self, assessor):
        report = assessor.assess(_minimal_df())
        assert report.module_readiness["contract_compliance"]["can_run"] is False

    def test_lower_overall_score(self, assessor):
        report = assessor.assess(_minimal_df())
        assert report.overall_readiness_score < 0.70

    def test_issues_list_not_empty(self, assessor):
        report = assessor.assess(_minimal_df())
        assert len(report.data_quality_issues) > 0

    def test_duplicate_effectiveness_without_line_items(self, assessor):
        report = assessor.assess(_minimal_df())
        dup = report.module_readiness["duplicate_detection"]
        assert dup["effectiveness"] == 0.75


# ==================================================================
# Missing critical fields
# ==================================================================

class TestMissingCriticalFields:
    def test_no_invoice_number(self, assessor):
        df = pd.DataFrame({
            "vendor_name": ["A", "B"],
            "invoice_date": [date(2024, 1, 1), date(2024, 6, 1)],
            "total_amount": [100, 200],
        })
        report = assessor.assess(df)
        assert report.module_readiness["duplicate_detection"]["can_run"] is False
        assert any("invoice_number" in i for i in report.data_quality_issues)

    def test_no_vendor(self, assessor):
        df = pd.DataFrame({
            "invoice_number": ["1", "2"],
            "invoice_date": [date(2024, 1, 1), date(2024, 6, 1)],
            "total_amount": [100, 200],
        })
        report = assessor.assess(df)
        assert report.module_readiness["duplicate_detection"]["can_run"] is False

    def test_no_amount(self, assessor):
        df = pd.DataFrame({
            "invoice_number": ["1"],
            "vendor_name": ["A"],
            "invoice_date": [date(2024, 1, 1)],
        })
        report = assessor.assess(df)
        assert report.module_readiness["split_invoicing"]["can_run"] is False
        assert any("total_amount" in i for i in report.data_quality_issues)

    def test_no_date(self, assessor):
        df = pd.DataFrame({
            "invoice_number": ["1"],
            "vendor_name": ["A"],
            "total_amount": [100],
        })
        report = assessor.assess(df)
        assert report.module_readiness["vendor_behavior"]["can_run"] is False
        assert report.date_range["earliest"] is None

    def test_partial_completeness_issue(self, assessor):
        df = pd.DataFrame({
            "invoice_number": ["1", "2", "", "4"],
            "vendor_name": ["A", "B", "C", "D"],
            "invoice_date": [date(2024, m, 1) for m in [1, 3, 5, 7]],
            "total_amount": [100, 200, 300, 400],
        })
        report = assessor.assess(df)
        assert any("invoice_number" in i and "empty" in i
                    for i in report.data_quality_issues)


# ==================================================================
# Short date ranges
# ==================================================================

class TestShortDateRange:
    def test_four_months_blocks_behavior(self, assessor):
        df = _minimal_df(n=10, months=4)
        report = assessor.assess(df)
        assert report.module_readiness["vendor_behavior"]["can_run"] is False
        assert any("months" in i.lower() for i in report.data_quality_issues)

    def test_six_months_allows_behavior(self, assessor):
        # months=7 → dates in months 1..7 → span of 6 calendar months
        df = _minimal_df(n=10, months=7)
        report = assessor.assess(df)
        assert report.module_readiness["vendor_behavior"]["can_run"] is True
        assert report.module_readiness["vendor_behavior"]["effectiveness"] == 0.60

    def test_twelve_months_higher_effectiveness(self, assessor):
        # Explicitly build dates spanning 12 months
        rows = [
            {"invoice_number": f"I{i}", "vendor_name": f"V{i%3}",
             "invoice_date": date(2024, 1, 15), "total_amount": Decimal("100")}
            for i in range(5)
        ] + [
            {"invoice_number": f"J{i}", "vendor_name": f"V{i%3}",
             "invoice_date": date(2025, 1, 15), "total_amount": Decimal("100")}
            for i in range(5)
        ]
        df = pd.DataFrame(rows)
        report = assessor.assess(df)
        behav = report.module_readiness["vendor_behavior"]
        assert behav["can_run"] is True
        assert behav["effectiveness"] == 0.85

    def test_one_month(self, assessor):
        df = pd.DataFrame({
            "invoice_number": ["1", "2", "3"],
            "vendor_name": ["A", "B", "C"],
            "invoice_date": [date(2024, 3, 1), date(2024, 3, 10), date(2024, 3, 20)],
            "total_amount": [100, 200, 300],
        })
        report = assessor.assess(df)
        assert report.date_range["months"] == 0
        assert report.module_readiness["vendor_behavior"]["can_run"] is False


# ==================================================================
# Empty dataset
# ==================================================================

class TestEmptyDataset:
    def test_empty_df_with_columns(self, assessor):
        """Columns present but zero rows — reports the empty-data issue."""
        df = pd.DataFrame(columns=[
            "invoice_number", "vendor_name", "invoice_date", "total_amount",
        ])
        report = assessor.assess(df)
        assert report.total_invoices == 0
        assert report.unique_vendors == 0
        assert any("empty" in i.lower() for i in report.data_quality_issues)

    def test_no_columns_no_modules(self, assessor):
        """Completely empty DataFrame — no module can run."""
        df = pd.DataFrame()
        report = assessor.assess(df)
        assert report.overall_readiness_score == 0.0
        for info in report.module_readiness.values():
            assert info["can_run"] is False


# ==================================================================
# Data type validation issues
# ==================================================================

class TestDataTypeValidation:
    def test_unparseable_dates(self, assessor):
        df = pd.DataFrame({
            "invoice_number": ["1", "2", "3"],
            "vendor_name": ["A", "B", "C"],
            "invoice_date": ["2024-01-01", "not-a-date", "2024-03-01"],
            "total_amount": ["100", "200", "300"],
        })
        report = assessor.assess(df)
        assert any("unparseable" in i for i in report.data_quality_issues)

    def test_non_numeric_amounts(self, assessor):
        df = pd.DataFrame({
            "invoice_number": ["1", "2"],
            "vendor_name": ["A", "B"],
            "invoice_date": [date(2024, 1, 1), date(2024, 6, 1)],
            "total_amount": ["abc", "200"],
        })
        report = assessor.assess(df)
        assert any("non-numeric" in i for i in report.data_quality_issues)

    def test_negative_amounts(self, assessor):
        df = pd.DataFrame({
            "invoice_number": ["1", "2"],
            "vendor_name": ["A", "B"],
            "invoice_date": [date(2024, 1, 1), date(2024, 6, 1)],
            "total_amount": ["-100", "200"],
        })
        report = assessor.assess(df)
        assert any("negative" in i for i in report.data_quality_issues)


# ==================================================================
# Supplementary data impact
# ==================================================================

class TestSupplementaryDataImpact:
    def test_vendor_master_improves_collusion(self, assessor):
        df = _minimal_df()
        r_no_vm = assessor.assess(df)
        r_with_vm = assessor.assess(
            df, vendor_df=pd.DataFrame({"vendor_id": ["V001"]}),
        )
        assert (
            r_with_vm.module_readiness["vendor_collusion"]["effectiveness"]
            > r_no_vm.module_readiness["vendor_collusion"]["effectiveness"]
        )

    def test_contracts_enable_module5(self, assessor):
        df = _minimal_df()
        r_no = assessor.assess(df)
        r_yes = assessor.assess(df, contracts=[_make_contract()])
        assert r_no.module_readiness["contract_compliance"]["can_run"] is False
        assert r_yes.module_readiness["contract_compliance"]["can_run"] is True

    def test_receipts_improve_phantom(self, assessor):
        df = _full_df()
        r_no = assessor.assess(df)
        r_yes = assessor.assess(
            df, receipts_df=pd.DataFrame({"receipt_id": ["R1"]}),
        )
        assert (
            r_yes.module_readiness["phantom_services"]["effectiveness"]
            >= r_no.module_readiness["phantom_services"]["effectiveness"]
        )

    def test_thresholds_improve_split(self, assessor):
        df = _minimal_df()
        r_no = assessor.assess(df)
        r_yes = assessor.assess(
            df, approval_thresholds=[{"level": "Mgr", "max_amount": 5000}],
        )
        assert (
            r_yes.module_readiness["split_invoicing"]["effectiveness"]
            > r_no.module_readiness["split_invoicing"]["effectiveness"]
        )


# ==================================================================
# print_report (smoke test — just ensure it doesn't raise)
# ==================================================================

class TestPrintReport:
    def test_prints_without_error(self, assessor):
        report = assessor.assess(_full_df())
        assessor.print_report(report)  # should not raise

    def test_prints_empty_report(self, assessor):
        report = assessor.assess(pd.DataFrame())
        assessor.print_report(report)


# ==================================================================
# Overall readiness score
# ==================================================================

class TestOverallScore:
    def test_weights_sum_to_one(self):
        from src.ingestion.data_quality import _MODULE_WEIGHTS
        assert sum(_MODULE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_score_range(self, assessor):
        report = assessor.assess(_full_df())
        assert 0.0 <= report.overall_readiness_score <= 1.0

    def test_zero_when_nothing_runs(self, assessor):
        report = assessor.assess(pd.DataFrame())
        assert report.overall_readiness_score == 0.0
