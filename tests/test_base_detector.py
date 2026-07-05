"""Tests for src.detection.base_detector — BaseDetector and ModuleRunner."""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.detection.base_detector import BaseDetector, ModuleRunner
from src.normalization.cache import CacheManager
from src.utils.constants import Finding, ModuleName, Severity


# ------------------------------------------------------------------
# Concrete test stubs
# ------------------------------------------------------------------

class StubDetectorA(BaseDetector):
    """Minimal concrete detector for testing."""

    def get_module_name(self) -> ModuleName:
        return ModuleName.DUPLICATE_DETECTION

    def get_required_fields(self) -> list[str]:
        return ["invoice_number", "vendor_name", "invoice_date", "total_amount"]

    def get_optional_fields(self) -> list[str]:
        return ["line_item_description", "po_number"]

    def detect(
        self,
        invoice_data: pd.DataFrame,
        vendor_data: Optional[pd.DataFrame] = None,
        supplementary_data: Optional[dict] = None,
    ) -> list[Finding]:
        self.stats["invoices_analyzed"] = len(invoice_data)
        # Produce one finding per row with amount > 9000 (for testing)
        for _, row in invoice_data.iterrows():
            if float(row.get("total_amount", 0)) > 9000:
                self.create_finding(
                    severity=Severity.CRITICAL,
                    confidence=0.95,
                    vendor_id=str(row.get("vendor_id", "")),
                    vendor_name=str(row.get("vendor_name", "")),
                    invoice_ids=[str(row.get("invoice_number", ""))],
                    amount_at_risk=Decimal(str(row["total_amount"])),
                    description="Stub finding for testing",
                    evidence={"stub": True},
                    recommended_action="Review",
                )
        return self.findings


class StubDetectorB(BaseDetector):
    """Second stub with different requirements."""

    def get_module_name(self) -> ModuleName:
        return ModuleName.PRICE_CREEP

    def get_required_fields(self) -> list[str]:
        return ["unit_price", "line_item_description", "invoice_date", "vendor_name"]

    def get_optional_fields(self) -> list[str]:
        return []

    def detect(
        self,
        invoice_data: pd.DataFrame,
        vendor_data: Optional[pd.DataFrame] = None,
        supplementary_data: Optional[dict] = None,
    ) -> list[Finding]:
        self.stats["invoices_analyzed"] = len(invoice_data)
        return self.findings


class FailingDetector(BaseDetector):
    """Detector that always raises."""

    def get_module_name(self) -> ModuleName:
        return ModuleName.PHANTOM_SERVICES

    def get_required_fields(self) -> list[str]:
        return ["vendor_name"]

    def get_optional_fields(self) -> list[str]:
        return []

    def detect(self, invoice_data, vendor_data=None, supplementary_data=None):
        raise RuntimeError("Intentional failure for testing")


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    cm = CacheManager(db_path=str(tmp_path / "test.db"))
    yield cm
    cm.close()


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def config():
    return {
        "duplicate_detection": {"fuzzy_composite_threshold": 0.85},
        "price_creep": {"min_data_points": 4},
    }


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "invoice_number": ["INV-001", "INV-002", "INV-003"],
        "vendor_id": ["V001", "V002", "V003"],
        "vendor_name": ["Acme", "Beta", "Gamma"],
        "invoice_date": ["2025-01-01", "2025-02-01", "2025-03-01"],
        "total_amount": [5000.0, 10000.0, 3000.0],
        "line_item_description": ["Widget", "Service", "Supply"],
        "po_number": ["PO-1", "PO-2", ""],
    })


# ==================================================================
# BaseDetector — abstract enforcement
# ==================================================================

class TestAbstractEnforcement:
    def test_cannot_instantiate_abc(self, config, cache, mock_llm):
        with pytest.raises(TypeError):
            BaseDetector(config, cache, mock_llm)


# ==================================================================
# can_run
# ==================================================================

class TestCanRun:
    def test_all_required_present(self, config, cache, mock_llm):
        det = StubDetectorA(config, cache, mock_llm)
        can, eff = det.can_run([
            "invoice_number", "vendor_name", "invoice_date", "total_amount",
        ])
        assert can is True
        assert eff == 0.6  # no optional fields present

    def test_all_fields_present(self, config, cache, mock_llm):
        det = StubDetectorA(config, cache, mock_llm)
        can, eff = det.can_run([
            "invoice_number", "vendor_name", "invoice_date", "total_amount",
            "line_item_description", "po_number",
        ])
        assert can is True
        assert eff == 1.0

    def test_one_optional_present(self, config, cache, mock_llm):
        det = StubDetectorA(config, cache, mock_llm)
        can, eff = det.can_run([
            "invoice_number", "vendor_name", "invoice_date", "total_amount",
            "line_item_description",
        ])
        assert can is True
        assert eff == 0.8  # 0.6 + 0.4*(1/2)

    def test_missing_required(self, config, cache, mock_llm):
        det = StubDetectorA(config, cache, mock_llm)
        can, eff = det.can_run(["invoice_number", "vendor_name"])
        assert can is False
        assert eff == 0.0

    def test_no_optional_fields(self, config, cache, mock_llm):
        det = StubDetectorB(config, cache, mock_llm)
        can, eff = det.can_run([
            "unit_price", "line_item_description", "invoice_date", "vendor_name",
        ])
        assert can is True
        assert eff == 1.0

    def test_extra_columns_ignored(self, config, cache, mock_llm):
        det = StubDetectorA(config, cache, mock_llm)
        can, eff = det.can_run([
            "invoice_number", "vendor_name", "invoice_date", "total_amount",
            "random_column", "another_one",
        ])
        assert can is True


# ==================================================================
# create_finding
# ==================================================================

class TestCreateFinding:
    def test_creates_valid_finding(self, config, cache, mock_llm):
        det = StubDetectorA(config, cache, mock_llm)
        finding = det.create_finding(
            severity=Severity.CRITICAL,
            confidence=0.95,
            vendor_id="V001",
            vendor_name="Acme",
            invoice_ids=["INV-001"],
            amount_at_risk=Decimal("5000.00"),
            description="Test finding",
            evidence={"key": "value"},
            recommended_action="Review invoice",
        )
        assert isinstance(finding, Finding)
        assert finding.module == ModuleName.DUPLICATE_DETECTION
        assert finding.severity == Severity.CRITICAL
        assert finding.confidence == 0.95
        assert finding.finding_id.startswith("SA-")
        assert isinstance(finding.created_at, datetime)
        assert finding.suppressed is False

    def test_finding_appended_to_list(self, config, cache, mock_llm):
        det = StubDetectorA(config, cache, mock_llm)
        assert len(det.findings) == 0
        det.create_finding(
            Severity.REVIEW, 0.8, "V1", "X", ["I1"],
            Decimal("100"), "desc", {}, "act",
        )
        assert len(det.findings) == 1

    def test_stats_incremented(self, config, cache, mock_llm):
        det = StubDetectorA(config, cache, mock_llm)
        det.create_finding(
            Severity.INFORMATIONAL, 0.5, "V1", "X", ["I1"],
            Decimal("50"), "d", {}, "a",
        )
        det.create_finding(
            Severity.CRITICAL, 0.9, "V2", "Y", ["I2"],
            Decimal("99"), "d", {}, "a",
        )
        assert det.stats["findings_generated"] == 2

    def test_unique_finding_ids(self, config, cache, mock_llm):
        det = StubDetectorA(config, cache, mock_llm)
        f1 = det.create_finding(
            Severity.REVIEW, 0.7, "V1", "X", ["I1"],
            Decimal("1"), "d", {}, "a",
        )
        f2 = det.create_finding(
            Severity.REVIEW, 0.7, "V1", "X", ["I1"],
            Decimal("1"), "d", {}, "a",
        )
        assert f1.finding_id != f2.finding_id


# ==================================================================
# detect
# ==================================================================

class TestDetect:
    def test_stub_produces_findings(self, config, cache, mock_llm, sample_df):
        det = StubDetectorA(config, cache, mock_llm)
        findings = det.detect(sample_df)
        # Only INV-002 has amount > 9000
        assert len(findings) == 1
        assert findings[0].vendor_name == "Beta"
        assert findings[0].amount_at_risk == Decimal("10000.0")

    def test_stats_updated(self, config, cache, mock_llm, sample_df):
        det = StubDetectorA(config, cache, mock_llm)
        det.detect(sample_df)
        stats = det.get_stats()
        assert stats["invoices_analyzed"] == 3
        assert stats["findings_generated"] == 1
        assert stats["module"] == "duplicate_detection"


# ==================================================================
# get_stats
# ==================================================================

class TestGetStats:
    def test_initial_stats(self, config, cache, mock_llm):
        det = StubDetectorA(config, cache, mock_llm)
        stats = det.get_stats()
        assert stats["invoices_analyzed"] == 0
        assert stats["findings_generated"] == 0
        assert stats["module"] == "duplicate_detection"
        assert "tier1_calls" in stats


# ==================================================================
# ModuleRunner
# ==================================================================

class TestModuleRunner:
    def test_register_module(self, config, cache, mock_llm):
        runner = ModuleRunner(config, cache, mock_llm)
        runner.register_module(StubDetectorA)
        assert len(runner.modules) == 1

    def test_register_multiple(self, config, cache, mock_llm):
        runner = ModuleRunner(config, cache, mock_llm)
        runner.register_module(StubDetectorA)
        runner.register_module(StubDetectorB)
        assert len(runner.modules) == 2

    def test_module_gets_specific_config(self, config, cache, mock_llm):
        runner = ModuleRunner(config, cache, mock_llm)
        runner.register_module(StubDetectorA)
        module = runner.modules[0]
        assert module.config == config["duplicate_detection"]

    def test_run_all_collects_findings(self, config, cache, mock_llm, sample_df):
        runner = ModuleRunner(config, cache, mock_llm)
        runner.register_module(StubDetectorA)
        findings = runner.run_all(sample_df)
        assert len(findings) == 1
        assert findings[0].module == ModuleName.DUPLICATE_DETECTION

    def test_skips_module_missing_fields(self, config, cache, mock_llm):
        runner = ModuleRunner(config, cache, mock_llm)
        runner.register_module(StubDetectorB)  # needs unit_price etc.
        df = pd.DataFrame({"vendor_name": ["A"], "invoice_date": ["2025-01-01"]})
        findings = runner.run_all(df)
        assert len(findings) == 0

    def test_run_all_multiple_modules(self, config, cache, mock_llm, sample_df):
        # Add unit_price so both can run
        sample_df["unit_price"] = [100.0, 200.0, 50.0]
        runner = ModuleRunner(config, cache, mock_llm)
        runner.register_module(StubDetectorA)
        runner.register_module(StubDetectorB)
        findings = runner.run_all(sample_df)
        # StubA produces 1 finding, StubB produces 0
        assert len(findings) == 1

    def test_failing_module_does_not_crash_runner(self, config, cache, mock_llm):
        runner = ModuleRunner(config, cache, mock_llm)
        runner.register_module(FailingDetector)
        df = pd.DataFrame({"vendor_name": ["A"]})
        findings = runner.run_all(df)
        assert len(findings) == 0

    def test_get_all_stats(self, config, cache, mock_llm, sample_df):
        runner = ModuleRunner(config, cache, mock_llm)
        runner.register_module(StubDetectorA)
        runner.run_all(sample_df)
        stats = runner.get_all_stats()
        assert stats["total_findings"] == 1
        assert stats["total_processing_time_seconds"] >= 0
        assert len(stats["modules"]) == 1
        assert stats["modules"][0]["module"] == "duplicate_detection"

    def test_processing_time_tracked(self, config, cache, mock_llm, sample_df):
        runner = ModuleRunner(config, cache, mock_llm)
        runner.register_module(StubDetectorA)
        runner.run_all(sample_df)
        stats = runner.modules[0].get_stats()
        assert stats["processing_time_seconds"] >= 0
        # Verify the field was actually set (not the default)
        assert "processing_time_seconds" in stats


# ==================================================================
# Edge cases
# ==================================================================

class TestEdgeCases:
    def test_empty_dataframe(self, config, cache, mock_llm):
        det = StubDetectorA(config, cache, mock_llm)
        df = pd.DataFrame(columns=[
            "invoice_number", "vendor_name", "invoice_date", "total_amount",
        ])
        findings = det.detect(df)
        assert findings == []

    def test_runner_no_modules(self, config, cache, mock_llm, sample_df):
        runner = ModuleRunner(config, cache, mock_llm)
        findings = runner.run_all(sample_df)
        assert findings == []

    def test_supplementary_data_passed_through(self, config, cache, mock_llm, sample_df):
        """Verify supplementary_data reaches the detect method."""
        class SpyDetector(StubDetectorA):
            received_supplementary = None
            def detect(self, invoice_data, vendor_data=None, supplementary_data=None):
                SpyDetector.received_supplementary = supplementary_data
                return []

        runner = ModuleRunner(config, cache, mock_llm)
        runner.register_module(SpyDetector)
        supp = {"contracts": [{"vendor_id": "V001"}]}
        runner.run_all(sample_df, supplementary_data=supp)
        assert SpyDetector.received_supplementary == supp

    def test_vendor_data_passed_through(self, config, cache, mock_llm, sample_df):
        class SpyDetector(StubDetectorA):
            received_vendor = None
            def detect(self, invoice_data, vendor_data=None, supplementary_data=None):
                SpyDetector.received_vendor = vendor_data
                return []

        runner = ModuleRunner(config, cache, mock_llm)
        runner.register_module(SpyDetector)
        vdf = pd.DataFrame({"vendor_id": ["V001"], "vendor_name": ["Acme"]})
        runner.run_all(sample_df, vendor_data=vdf)
        assert SpyDetector.received_vendor is not None
        assert len(SpyDetector.received_vendor) == 1
