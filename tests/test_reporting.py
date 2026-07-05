"""Tests for src.reporting — ReportGenerator and DashboardDataGenerator."""

import json
import os
from decimal import Decimal

import pytest

from src.reporting.dashboard_data import DashboardDataGenerator
from src.reporting.report_generator import ReportGenerator
from src.utils.constants import (
    DataQualityReport,
    Finding,
    ModuleName,
    RiskTier,
    Severity,
    generate_finding_id,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _f(
    module: ModuleName = ModuleName.DUPLICATE_DETECTION,
    severity: Severity = Severity.REVIEW,
    confidence: float = 0.80,
    vendor_id: str = "V001",
    vendor_name: str = "Acme",
    amount: float = 5000.0,
    description: str = "Test finding",
    suppressed: bool = False,
) -> Finding:
    f = Finding(
        finding_id=generate_finding_id(),
        module=module,
        severity=severity,
        confidence=confidence,
        vendor_id=vendor_id,
        vendor_name=vendor_name,
        invoice_ids=["INV-001"],
        amount_at_risk=Decimal(str(amount)),
        description=description,
        evidence={"method": "test"},
        recommended_action="Review",
    )
    f.suppressed = suppressed
    if suppressed:
        f.suppression_reason = "test"
    return f


def _vendor_scores() -> dict:
    return {
        "V001": {
            "score": 85.0, "tier": RiskTier.CRITICAL,
            "contributing_factors": ["duplicate_detection (2 findings)"],
            "findings_count": 3, "amount_at_risk": 15000.0,
            "vendor_name": "Acme",
        },
        "V002": {
            "score": 30.0, "tier": RiskTier.MEDIUM,
            "contributing_factors": ["price_creep (1 finding)"],
            "findings_count": 1, "amount_at_risk": 2000.0,
            "vendor_name": "Beta",
        },
    }


def _invoice_scores() -> dict:
    return {
        "INV-001": {
            "score": 70.0, "tier": RiskTier.HIGH,
            "findings": ["SA-001"], "vendor_id": "V001", "vendor_name": "Acme",
        },
    }


def _alert_summary() -> dict:
    return {
        "critical_count": 2,
        "review_count": 3,
        "informational_count": 1,
        "total_amount_at_risk": Decimal("25000"),
        "critical_amount": Decimal("15000"),
        "top_critical_alerts": [],
        "modules_triggered": ["duplicate_detection", "price_creep"],
        "vendors_flagged": 2,
    }


def _dq_report() -> DataQualityReport:
    return DataQualityReport(
        total_invoices=5000,
        date_range={"earliest": "2024-01-01", "latest": "2025-06-30"},
        unique_vendors=140,
        fields_present={},
        module_readiness={},
        data_quality_issues=["10% dates unparseable"],
        overall_readiness_score=0.78,
    )


def _sample_findings() -> list[Finding]:
    return [
        _f(severity=Severity.CRITICAL, amount=10000, vendor_id="V001"),
        _f(severity=Severity.CRITICAL, amount=5000, vendor_id="V001",
           module=ModuleName.PRICE_CREEP),
        _f(severity=Severity.REVIEW, amount=3000, vendor_id="V002",
           vendor_name="Beta", module=ModuleName.PHANTOM_SERVICES),
        _f(severity=Severity.INFORMATIONAL, amount=500, vendor_id="V003",
           vendor_name="Gamma"),
        _f(severity=Severity.REVIEW, amount=2000, suppressed=True),
    ]


# ==================================================================
# ReportGenerator
# ==================================================================

class TestReportGenerator:
    @pytest.fixture
    def gen(self, tmp_path):
        return ReportGenerator(str(tmp_path / "reports"))

    def test_generate_all_reports(self, gen):
        paths = gen.generate_all_reports(
            _sample_findings(), _vendor_scores(),
            _invoice_scores(), _alert_summary(), _dq_report(),
        )
        assert len(paths) == 8  # 2 per report type × 4
        for p in paths:
            assert os.path.exists(p)

    def test_executive_summary_json(self, gen):
        paths = gen.generate_executive_summary(
            _sample_findings(), _vendor_scores(),
            _alert_summary(), _dq_report(),
        )
        json_path = [p for p in paths if p.endswith(".json")][0]
        with open(json_path) as f:
            data = json.load(f)
        assert "total_amount_at_risk" in data
        assert "findings_by_severity" in data
        assert data["data_quality"]["total_invoices"] == 5000

    def test_executive_summary_html(self, gen):
        paths = gen.generate_executive_summary(
            _sample_findings(), _vendor_scores(), _alert_summary(), None,
        )
        html_path = [p for p in paths if p.endswith(".html")][0]
        with open(html_path) as f:
            content = f.read()
        assert "<html>" in content
        assert "CRITICAL" in content

    def test_detailed_findings_json(self, gen):
        paths = gen.generate_detailed_findings(_sample_findings())
        json_path = [p for p in paths if p.endswith(".json")][0]
        with open(json_path) as f:
            data = json.load(f)
        # 4 active findings (1 suppressed excluded)
        assert len(data) == 4

    def test_detailed_findings_csv(self, gen):
        paths = gen.generate_detailed_findings(_sample_findings())
        csv_path = [p for p in paths if p.endswith(".csv")][0]
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) >= 2  # header + data

    def test_vendor_risk_report(self, gen):
        paths = gen.generate_vendor_risk_report(
            _vendor_scores(), _sample_findings(),
        )
        json_path = [p for p in paths if p.endswith(".json")][0]
        with open(json_path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["risk_score"] >= data[1]["risk_score"]

    def test_vendor_risk_html(self, gen):
        paths = gen.generate_vendor_risk_report(
            _vendor_scores(), _sample_findings(),
        )
        html_path = [p for p in paths if p.endswith(".html")][0]
        with open(html_path) as f:
            content = f.read()
        assert "Acme" in content

    def test_recovery_report(self, gen):
        paths = gen.generate_recovery_report(_sample_findings())
        json_path = [p for p in paths if p.endswith(".json")][0]
        with open(json_path) as f:
            data = json.load(f)
        assert data["total_recoverable"] > 0
        assert data["gain_share_25pct"] > 0
        assert len(data["opportunities"]) > 0
        # Gain share = 25%
        assert data["gain_share_25pct"] == pytest.approx(
            data["total_recoverable"] * 0.25, rel=0.01,
        )

    def test_recovery_csv(self, gen):
        paths = gen.generate_recovery_report(_sample_findings())
        csv_path = [p for p in paths if p.endswith(".csv")][0]
        assert os.path.exists(csv_path)

    def test_empty_findings(self, gen):
        paths = gen.generate_all_reports([], {}, {}, _alert_summary())
        assert len(paths) == 8

    def test_output_dir_created(self, tmp_path):
        new_dir = str(tmp_path / "new" / "subdir")
        gen = ReportGenerator(new_dir)
        assert os.path.isdir(new_dir)


# ==================================================================
# DashboardDataGenerator
# ==================================================================

class TestDashboardDataGenerator:
    @pytest.fixture
    def gen(self):
        return DashboardDataGenerator()

    def test_generates_all_sections(self, gen):
        data = gen.generate_dashboard_json(
            _sample_findings(), _vendor_scores(),
            _invoice_scores(), _alert_summary(), _dq_report(),
        )
        for key in ["summary", "risk_distribution", "findings_by_module",
                     "amount_by_module", "top_vendors", "top_findings",
                     "timeline", "data_quality"]:
            assert key in data, f"Missing key: {key}"

    def test_risk_distribution(self, gen):
        data = gen.generate_dashboard_json(
            _sample_findings(), _vendor_scores(),
            _invoice_scores(), _alert_summary(),
        )
        dist = data["risk_distribution"]
        assert "critical" in dist
        assert sum(dist.values()) == 2  # 2 vendor scores

    def test_findings_by_module(self, gen):
        data = gen.generate_dashboard_json(
            _sample_findings(), _vendor_scores(),
            _invoice_scores(), _alert_summary(),
        )
        by_mod = data["findings_by_module"]
        assert "duplicate_detection" in by_mod

    def test_amount_by_module(self, gen):
        data = gen.generate_dashboard_json(
            _sample_findings(), _vendor_scores(),
            _invoice_scores(), _alert_summary(),
        )
        by_mod = data["amount_by_module"]
        assert sum(by_mod.values()) > 0

    def test_top_vendors(self, gen):
        data = gen.generate_dashboard_json(
            _sample_findings(), _vendor_scores(),
            _invoice_scores(), _alert_summary(),
        )
        assert len(data["top_vendors"]) == 2
        assert data["top_vendors"][0]["score"] >= data["top_vendors"][1]["score"]

    def test_top_findings(self, gen):
        data = gen.generate_dashboard_json(
            _sample_findings(), _vendor_scores(),
            _invoice_scores(), _alert_summary(),
        )
        assert len(data["top_findings"]) <= 20
        assert all("finding_id" in f for f in data["top_findings"])

    def test_timeline(self, gen):
        data = gen.generate_dashboard_json(
            _sample_findings(), _vendor_scores(),
            _invoice_scores(), _alert_summary(),
        )
        assert isinstance(data["timeline"], list)

    def test_data_quality_present(self, gen):
        data = gen.generate_dashboard_json(
            _sample_findings(), {}, {}, _alert_summary(), _dq_report(),
        )
        assert data["data_quality"]["total_invoices"] == 5000

    def test_data_quality_none(self, gen):
        data = gen.generate_dashboard_json(
            _sample_findings(), {}, {}, _alert_summary(),
        )
        assert data["data_quality"] is None

    def test_writes_to_file(self, gen, tmp_path):
        out = str(tmp_path / "dash" / "dashboard_data.json")
        data = gen.generate_dashboard_json(
            _sample_findings(), _vendor_scores(),
            _invoice_scores(), _alert_summary(),
            output_path=out,
        )
        assert os.path.exists(out)
        with open(out) as f:
            loaded = json.load(f)
        assert loaded["summary"]["critical_count"] == 2

    def test_empty_findings(self, gen):
        data = gen.generate_dashboard_json([], {}, {}, _alert_summary())
        assert data["findings_by_module"] == {}
        assert data["top_findings"] == []
