"""Tests for src.adjudication.risk_scorer — RiskScorer."""

from decimal import Decimal

import pytest

from src.adjudication.risk_scorer import RiskScorer
from src.utils.constants import (
    Finding,
    ModuleName,
    RiskTier,
    Severity,
    generate_finding_id,
)


def _f(
    module: ModuleName = ModuleName.DUPLICATE_DETECTION,
    severity: Severity = Severity.REVIEW,
    confidence: float = 0.80,
    vendor_id: str = "V001",
    vendor_name: str = "Acme",
    invoice_ids: list[str] = None,
    amount: float = 5000.0,
    suppressed: bool = False,
) -> Finding:
    f = Finding(
        finding_id=generate_finding_id(),
        module=module,
        severity=severity,
        confidence=confidence,
        vendor_id=vendor_id,
        vendor_name=vendor_name,
        invoice_ids=invoice_ids or ["INV-001"],
        amount_at_risk=Decimal(str(amount)),
        description="Test",
        evidence={},
        recommended_action="Review",
    )
    f.suppressed = suppressed
    if suppressed:
        f.suppression_reason = "test"
    return f


@pytest.fixture
def scorer():
    return RiskScorer()


# ==================================================================
# Vendor scoring
# ==================================================================

class TestScoreVendors:
    def test_single_critical_finding(self, scorer):
        findings = [
            _f(severity=Severity.CRITICAL, confidence=0.95,
               module=ModuleName.VENDOR_COLLUSION),
        ]
        scores = scorer.score_vendors(findings)
        assert "V001" in scores
        s = scores["V001"]
        assert 0 < s["score"] <= 100
        assert s["findings_count"] == 1
        assert s["vendor_name"] == "Acme"

    def test_higher_severity_higher_score(self, scorer):
        crit = [_f(severity=Severity.CRITICAL, confidence=0.90)]
        info = [_f(severity=Severity.INFORMATIONAL, confidence=0.90)]
        sc = scorer.score_vendors(crit)["V001"]["score"]
        si = scorer.score_vendors(info)["V001"]["score"]
        assert sc > si

    def test_higher_confidence_higher_score(self, scorer):
        hi = [_f(confidence=0.95)]
        lo = [_f(confidence=0.30)]
        sh = scorer.score_vendors(hi)["V001"]["score"]
        sl = scorer.score_vendors(lo)["V001"]["score"]
        assert sh > sl

    def test_multiple_findings_accumulate(self, scorer):
        one = [_f()]
        two = [_f(), _f(module=ModuleName.PRICE_CREEP)]
        s1 = scorer.score_vendors(one)["V001"]["score"]
        s2 = scorer.score_vendors(two)["V001"]["score"]
        assert s2 > s1

    def test_suppressed_excluded(self, scorer):
        findings = [
            _f(suppressed=True),
            _f(vendor_id="V002", vendor_name="Beta"),
        ]
        scores = scorer.score_vendors(findings)
        assert "V001" not in scores
        assert "V002" in scores

    def test_contributing_factors(self, scorer):
        findings = [
            _f(module=ModuleName.DUPLICATE_DETECTION),
            _f(module=ModuleName.DUPLICATE_DETECTION),
            _f(module=ModuleName.PRICE_CREEP),
        ]
        s = scorer.score_vendors(findings)["V001"]
        assert any("duplicate_detection" in f for f in s["contributing_factors"])
        assert any("price_creep" in f for f in s["contributing_factors"])

    def test_amount_at_risk_summed(self, scorer):
        findings = [_f(amount=1000), _f(amount=2000)]
        s = scorer.score_vendors(findings)["V001"]
        assert s["amount_at_risk"] == 3000.0

    def test_multiple_vendors(self, scorer):
        findings = [
            _f(vendor_id="V001"),
            _f(vendor_id="V002", vendor_name="Beta"),
        ]
        scores = scorer.score_vendors(findings)
        assert len(scores) == 2


# ==================================================================
# Risk tiers
# ==================================================================

class TestTiers:
    def test_critical_tier(self, scorer):
        # Many high-severity findings → CRITICAL
        findings = [
            _f(severity=Severity.CRITICAL, confidence=0.95,
               module=ModuleName.VENDOR_COLLUSION),
            _f(severity=Severity.CRITICAL, confidence=0.90,
               module=ModuleName.PHANTOM_SERVICES),
        ]
        s = scorer.score_vendors(findings)["V001"]
        assert s["tier"] == RiskTier.CRITICAL

    def test_low_tier(self, scorer):
        # Single info finding → LOW
        findings = [_f(severity=Severity.INFORMATIONAL, confidence=0.30)]
        s = scorer.score_vendors(findings)["V001"]
        assert s["tier"] == RiskTier.LOW

    def test_empty_findings(self, scorer):
        assert scorer.score_vendors([]) == {}


# ==================================================================
# Invoice scoring
# ==================================================================

class TestScoreInvoices:
    def test_basic(self, scorer):
        findings = [
            _f(invoice_ids=["INV-001", "INV-002"]),
        ]
        scores = scorer.score_invoices(findings)
        assert "INV-001" in scores
        assert "INV-002" in scores
        assert scores["INV-001"]["score"] > 0
        assert len(scores["INV-001"]["findings"]) == 1

    def test_multiple_findings_per_invoice(self, scorer):
        findings = [
            _f(invoice_ids=["INV-001"]),
            _f(invoice_ids=["INV-001"], module=ModuleName.PRICE_CREEP),
        ]
        scores = scorer.score_invoices(findings)
        assert len(scores["INV-001"]["findings"]) == 2

    def test_suppressed_excluded(self, scorer):
        findings = [_f(invoice_ids=["INV-001"], suppressed=True)]
        assert scorer.score_invoices(findings) == {}

    def test_includes_vendor_info(self, scorer):
        findings = [_f(invoice_ids=["INV-001"])]
        s = scorer.score_invoices(findings)["INV-001"]
        assert s["vendor_id"] == "V001"
        assert s["vendor_name"] == "Acme"


# ==================================================================
# Top-N queries
# ==================================================================

class TestTopN:
    def test_top_vendors(self, scorer):
        findings = [
            _f(vendor_id="V001", severity=Severity.CRITICAL, confidence=0.95),
            _f(vendor_id="V002", vendor_name="Beta",
               severity=Severity.INFORMATIONAL, confidence=0.30),
        ]
        scores = scorer.score_vendors(findings)
        top = scorer.get_top_risk_vendors(scores, n=1)
        assert len(top) == 1
        assert top[0]["vendor_id"] == "V001"

    def test_top_invoices(self, scorer):
        findings = [
            _f(invoice_ids=["INV-H"], severity=Severity.CRITICAL, confidence=0.95),
            _f(invoice_ids=["INV-L"], severity=Severity.INFORMATIONAL, confidence=0.30),
        ]
        scores = scorer.score_invoices(findings)
        top = scorer.get_top_risk_invoices(scores, n=1)
        assert len(top) == 1
        assert top[0]["invoice_id"] == "INV-H"

    def test_top_n_larger_than_total(self, scorer):
        findings = [_f()]
        scores = scorer.score_vendors(findings)
        top = scorer.get_top_risk_vendors(scores, n=100)
        assert len(top) == 1


# ==================================================================
# Risk distribution
# ==================================================================

class TestRiskDistribution:
    def test_distribution(self, scorer):
        findings = [
            _f(vendor_id="V001", severity=Severity.CRITICAL, confidence=0.95,
               module=ModuleName.VENDOR_COLLUSION),
            _f(vendor_id="V001", severity=Severity.CRITICAL, confidence=0.90,
               module=ModuleName.PHANTOM_SERVICES),
            _f(vendor_id="V002", vendor_name="Beta",
               severity=Severity.INFORMATIONAL, confidence=0.30),
        ]
        scores = scorer.score_vendors(findings)
        dist = scorer.get_risk_distribution(scores)
        assert dist["critical"] >= 1 or dist["high"] >= 1
        assert dist["low"] >= 0
        total = sum(dist.values())
        assert total == len(scores)

    def test_empty(self, scorer):
        dist = scorer.get_risk_distribution({})
        assert all(v == 0 for v in dist.values())


# ==================================================================
# Module weights
# ==================================================================

class TestModuleWeights:
    def test_collusion_weighted_higher(self, scorer):
        """Collusion (w=0.20) should score higher than market (w=0.05)
        for same severity and confidence."""
        col = [_f(module=ModuleName.VENDOR_COLLUSION,
                  severity=Severity.CRITICAL, confidence=0.90)]
        mkt = [_f(module=ModuleName.MARKET_PRICE,
                  severity=Severity.CRITICAL, confidence=0.90)]
        sc = scorer.score_vendors(col)["V001"]["score"]
        sm = scorer.score_vendors(mkt)["V001"]["score"]
        assert sc > sm

    def test_weights_sum_to_one(self):
        total = sum(RiskScorer.MODULE_WEIGHTS.values())
        assert total == pytest.approx(1.0)


# ==================================================================
# Edge cases
# ==================================================================

class TestEdgeCases:
    def test_zero_confidence(self, scorer):
        findings = [_f(confidence=0.0)]
        scores = scorer.score_vendors(findings)
        assert scores["V001"]["score"] == 0.0

    def test_score_range(self, scorer):
        findings = [
            _f(severity=Severity.CRITICAL, confidence=1.0,
               module=m)
            for m in ModuleName
        ]
        scores = scorer.score_vendors(findings)
        assert 0 <= scores["V001"]["score"] <= 100
