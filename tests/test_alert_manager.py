"""Tests for src.reporting.alert_manager — AlertManager."""

from decimal import Decimal

import pytest

from src.reporting.alert_manager import AlertManager
from src.utils.constants import (
    Finding,
    ModuleName,
    Severity,
    generate_finding_id,
)


def _f(
    module: ModuleName = ModuleName.DUPLICATE_DETECTION,
    severity: Severity = Severity.REVIEW,
    confidence: float = 0.80,
    amount: float = 5000.0,
    vendor_id: str = "V001",
    vendor_name: str = "Acme",
    evidence: dict = None,
    suppressed: bool = False,
    correlations: list[str] = None,
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
        description="Test finding",
        evidence=evidence or {},
        recommended_action="Review",
        module_correlations=correlations or [],
    )
    f.suppressed = suppressed
    if suppressed:
        f.suppression_reason = "test"
    return f


@pytest.fixture
def mgr():
    return AlertManager(config={
        "alert_tiers": {
            "critical": {"min_confidence": 0.85, "min_amount": 5000},
            "review": {"min_confidence": 0.60, "min_amount": 1000},
            "informational": {"min_confidence": 0.30, "min_amount": 0},
        }
    })


# ==================================================================
# Categorization — CRITICAL
# ==================================================================

class TestCritical:
    def test_high_confidence_high_amount(self, mgr):
        f = _f(confidence=0.90, amount=10000, severity=Severity.REVIEW)
        result = mgr.categorize_alerts([f])
        assert f in result["critical"]

    def test_severity_already_critical(self, mgr):
        f = _f(severity=Severity.CRITICAL, confidence=0.50, amount=100)
        result = mgr.categorize_alerts([f])
        assert f in result["critical"]

    def test_bank_change_always_critical(self, mgr):
        f = _f(
            module=ModuleName.VENDOR_BEHAVIOR,
            severity=Severity.REVIEW,
            confidence=0.60, amount=500,
            evidence={"method": "bank_change"},
        )
        result = mgr.categorize_alerts([f])
        assert f in result["critical"]

    def test_multi_module_corroboration(self, mgr):
        f = _f(
            severity=Severity.REVIEW,
            confidence=0.60, amount=500,
            correlations=["id-a", "id-b"],
        )
        result = mgr.categorize_alerts([f])
        assert f in result["critical"]


# ==================================================================
# Categorization — REVIEW
# ==================================================================

class TestReview:
    def test_moderate_confidence_and_amount(self, mgr):
        f = _f(confidence=0.70, amount=3000, severity=Severity.REVIEW)
        result = mgr.categorize_alerts([f])
        assert f in result["review"]

    def test_high_confidence_low_amount(self, mgr):
        f = _f(confidence=0.90, amount=1000, severity=Severity.REVIEW)
        result = mgr.categorize_alerts([f])
        assert f in result["review"]


# ==================================================================
# Categorization — INFORMATIONAL
# ==================================================================

class TestInformational:
    def test_low_confidence(self, mgr):
        f = _f(confidence=0.40, amount=200, severity=Severity.INFORMATIONAL)
        result = mgr.categorize_alerts([f])
        assert f in result["informational"]

    def test_low_amount(self, mgr):
        f = _f(confidence=0.50, amount=50, severity=Severity.INFORMATIONAL)
        result = mgr.categorize_alerts([f])
        assert f in result["informational"]


# ==================================================================
# Suppressed findings excluded
# ==================================================================

class TestSuppressed:
    def test_suppressed_excluded(self, mgr):
        f = _f(suppressed=True, confidence=0.99, amount=99999)
        result = mgr.categorize_alerts([f])
        assert len(result["critical"]) == 0
        assert len(result["review"]) == 0
        assert len(result["informational"]) == 0


# ==================================================================
# Sorting
# ==================================================================

class TestSorting:
    def test_critical_sorted_by_amount_desc(self, mgr):
        f1 = _f(severity=Severity.CRITICAL, amount=10000)
        f2 = _f(severity=Severity.CRITICAL, amount=50000)
        f3 = _f(severity=Severity.CRITICAL, amount=1000)
        result = mgr.categorize_alerts([f1, f2, f3])
        amounts = [float(f.amount_at_risk) for f in result["critical"]]
        assert amounts == sorted(amounts, reverse=True)

    def test_tiebreak_by_confidence(self, mgr):
        f1 = _f(severity=Severity.CRITICAL, amount=5000, confidence=0.80)
        f2 = _f(severity=Severity.CRITICAL, amount=5000, confidence=0.95)
        result = mgr.categorize_alerts([f1, f2])
        confs = [f.confidence for f in result["critical"]]
        assert confs[0] >= confs[1]


# ==================================================================
# Alert summary
# ==================================================================

class TestAlertSummary:
    def test_summary_counts(self, mgr):
        findings = [
            _f(severity=Severity.CRITICAL, amount=10000),
            _f(severity=Severity.CRITICAL, amount=5000),
            _f(confidence=0.70, amount=3000, severity=Severity.REVIEW),
            _f(confidence=0.40, amount=200, severity=Severity.INFORMATIONAL),
        ]
        cats = mgr.categorize_alerts(findings)
        summary = mgr.generate_alert_summary(cats)
        assert summary["critical_count"] == 2
        assert summary["review_count"] == 1
        assert summary["informational_count"] == 1

    def test_summary_amounts(self, mgr):
        findings = [
            _f(severity=Severity.CRITICAL, amount=10000),
            _f(confidence=0.70, amount=3000, severity=Severity.REVIEW),
        ]
        cats = mgr.categorize_alerts(findings)
        summary = mgr.generate_alert_summary(cats)
        assert summary["total_amount_at_risk"] == Decimal("13000")
        assert summary["critical_amount"] == Decimal("10000")

    def test_top_critical(self, mgr):
        findings = [
            _f(severity=Severity.CRITICAL, amount=10000 + i)
            for i in range(7)
        ]
        cats = mgr.categorize_alerts(findings)
        summary = mgr.generate_alert_summary(cats)
        assert len(summary["top_critical_alerts"]) == 5

    def test_modules_triggered(self, mgr):
        findings = [
            _f(module=ModuleName.DUPLICATE_DETECTION, severity=Severity.CRITICAL),
            _f(module=ModuleName.PRICE_CREEP, severity=Severity.CRITICAL),
        ]
        cats = mgr.categorize_alerts(findings)
        summary = mgr.generate_alert_summary(cats)
        assert "duplicate_detection" in summary["modules_triggered"]
        assert "price_creep" in summary["modules_triggered"]

    def test_vendors_flagged(self, mgr):
        findings = [
            _f(vendor_id="V001", severity=Severity.CRITICAL),
            _f(vendor_id="V002", severity=Severity.CRITICAL),
            _f(vendor_id="V001", confidence=0.70, amount=2000, severity=Severity.REVIEW),
        ]
        cats = mgr.categorize_alerts(findings)
        summary = mgr.generate_alert_summary(cats)
        assert summary["vendors_flagged"] == 2

    def test_empty(self, mgr):
        cats = mgr.categorize_alerts([])
        summary = mgr.generate_alert_summary(cats)
        assert summary["critical_count"] == 0
        assert summary["total_amount_at_risk"] == Decimal("0")
        assert summary["vendors_flagged"] == 0


# ==================================================================
# Default config
# ==================================================================

class TestDefaultConfig:
    def test_works_without_config(self):
        mgr = AlertManager()
        f = _f(severity=Severity.CRITICAL, amount=10000)
        result = mgr.categorize_alerts([f])
        assert f in result["critical"]
