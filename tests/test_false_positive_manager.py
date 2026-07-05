"""Tests for src.adjudication.false_positive_manager."""

from decimal import Decimal

import pytest

from src.adjudication.false_positive_manager import (
    FalsePositiveManager,
    _pattern_hash,
)
from src.normalization.cache import CacheManager
from src.utils.constants import (
    FeedbackVerdict,
    Finding,
    ModuleName,
    Severity,
    generate_finding_id,
)


def _f(
    module: ModuleName = ModuleName.DUPLICATE_DETECTION,
    severity: Severity = Severity.REVIEW,
    confidence: float = 0.80,
    vendor_id: str = "V001",
    invoice_ids: list[str] = None,
    evidence: dict = None,
    suppressed: bool = False,
) -> Finding:
    f = Finding(
        finding_id=generate_finding_id(),
        module=module,
        severity=severity,
        confidence=confidence,
        vendor_id=vendor_id,
        vendor_name="Acme",
        invoice_ids=invoice_ids or ["INV-001"],
        amount_at_risk=Decimal("5000"),
        description="Test",
        evidence=evidence or {"method": "exact_match"},
        recommended_action="Review",
    )
    f.suppressed = suppressed
    if suppressed:
        f.suppression_reason = "pre-existing"
    return f


@pytest.fixture
def cache(tmp_path):
    cm = CacheManager(db_path=str(tmp_path / "test.db"))
    yield cm
    cm.close()


@pytest.fixture
def mgr(cache):
    return FalsePositiveManager(cache=cache)


# ==================================================================
# _pattern_hash
# ==================================================================

class TestPatternHash:
    def test_deterministic(self):
        f = _f(evidence={"method": "exact_match"})
        assert _pattern_hash(f) == _pattern_hash(f)

    def test_different_module(self):
        a = _f(module=ModuleName.DUPLICATE_DETECTION)
        b = _f(module=ModuleName.PRICE_CREEP)
        assert _pattern_hash(a) != _pattern_hash(b)

    def test_different_vendor(self):
        a = _f(vendor_id="V001")
        b = _f(vendor_id="V002")
        assert _pattern_hash(a) != _pattern_hash(b)

    def test_different_method(self):
        a = _f(evidence={"method": "exact_match"})
        b = _f(evidence={"method": "fuzzy_match"})
        assert _pattern_hash(a) != _pattern_hash(b)


# ==================================================================
# Record feedback
# ==================================================================

class TestRecordFeedback:
    def test_records_to_cache(self, mgr, cache):
        mgr.record_feedback("SA-test001", "CUST1",
                            FeedbackVerdict.FALSE_POSITIVE, "not fraud")
        fb = cache.get_feedback_for_pattern("", "")
        assert any(f["finding_id"] == "SA-test001" for f in fb)

    def test_records_confirmed_fraud(self, mgr, cache):
        mgr.record_feedback("SA-test002", "CUST1",
                            FeedbackVerdict.CONFIRMED_FRAUD)
        fb = cache.get_feedback_for_pattern("", "")
        match = [f for f in fb if f["finding_id"] == "SA-test002"]
        assert match[0]["verdict"] == "confirmed_fraud"


# ==================================================================
# Auto-suppression rules
# ==================================================================

class TestAutoSuppression:
    def test_suppress_after_3_customer_fps(self, mgr, cache):
        # Simulate 3 FP verdicts whose finding_ids contain the module name
        # so the LIKE-based cache query matches them.
        finding = _f(evidence={"method": "exact_match"})
        ph = _pattern_hash(finding)
        for i in range(3):
            fid = f"SA-{finding.module.value}-{ph}-{i}"
            cache.record_feedback(fid, "CUST1", "false_positive", None)

        mgr.apply_suppression_rules([finding], "CUST1")
        assert finding.suppressed is True
        assert "customer" in finding.suppression_reason.lower()

    def test_no_suppress_below_threshold(self, mgr, cache):
        # Only 2 FP verdicts — below threshold of 3
        for i in range(2):
            fid = f"SA-dup-V001-{i}"
            cache.record_feedback(fid, "CUST1", "false_positive", None)

        finding = _f(evidence={"method": "exact_match"})
        mgr.apply_suppression_rules([finding], "CUST1")
        assert finding.suppressed is False

    def test_suppress_global_pattern(self, mgr, cache):
        finding = _f(evidence={"method": "exact_match"})
        ph = _pattern_hash(finding)
        for i in range(3):
            fid = f"SA-{finding.module.value}-{ph}-g{i}"
            cache.record_feedback(fid, f"CUST{i}", "false_positive", None)

        mgr.apply_suppression_rules([finding], "CUST99")
        assert finding.suppressed is True
        assert "across customers" in finding.suppression_reason.lower()

    def test_already_suppressed_left_alone(self, mgr, cache):
        finding = _f(suppressed=True)
        original_reason = finding.suppression_reason
        mgr.apply_suppression_rules([finding], "CUST1")
        assert finding.suppression_reason == original_reason

    def test_confirmed_fraud_not_counted(self, mgr, cache):
        # 3 confirmed_fraud — should NOT trigger suppression
        for i in range(3):
            fid = f"SA-dup-V001-{i}"
            cache.record_feedback(fid, "CUST1", "confirmed_fraud", None)

        finding = _f(evidence={"method": "exact_match"})
        mgr.apply_suppression_rules([finding], "CUST1")
        assert finding.suppressed is False


# ==================================================================
# FP rate computation
# ==================================================================

class TestFalsePositiveRate:
    def test_basic_rate(self, mgr, cache):
        # 3 FP + 2 confirmed = 60% FP rate
        for i in range(3):
            cache.record_feedback(
                f"SA-dup-{i}", "CUST1", "false_positive", None)
        for i in range(2):
            cache.record_feedback(
                f"SA-dup-cf-{i}", "CUST1", "confirmed_fraud", None)

        rates = mgr.get_false_positive_rate("CUST1")
        dup = rates.get(ModuleName.DUPLICATE_DETECTION.value)
        assert dup is not None
        assert dup["total_reviewed"] == 5
        assert dup["false_positives"] == 3
        assert dup["fp_rate"] == 0.6

    def test_no_feedback_zero_rate(self, mgr):
        rates = mgr.get_false_positive_rate("NOBODY")
        for info in rates.values():
            assert info["fp_rate"] == 0.0

    def test_filter_by_module(self, mgr, cache):
        cache.record_feedback("SA-dup-0", "CUST1", "false_positive", None)
        rates = mgr.get_false_positive_rate(
            "CUST1", module=ModuleName.DUPLICATE_DETECTION)
        assert len(rates) == 1
        assert ModuleName.DUPLICATE_DETECTION.value in rates


# ==================================================================
# Common FP patterns
# ==================================================================

class TestCommonPatterns:
    def test_returns_patterns(self, mgr, cache):
        for i in range(5):
            cache.record_feedback(
                f"SA-dup-{i}", "CUST1", "false_positive", None)
        for i in range(2):
            cache.record_feedback(
                f"SA-price-{i}", "CUST1", "false_positive", None)

        patterns = mgr.get_common_fp_patterns("CUST1")
        assert len(patterns) >= 1
        # Duplicate should be most common
        assert patterns[0]["pattern_count"] >= 2

    def test_empty_when_no_fps(self, mgr):
        assert mgr.get_common_fp_patterns("NOBODY") == []

    def test_global_patterns(self, mgr, cache):
        for i in range(3):
            cache.record_feedback(
                f"SA-dup-{i}", f"CUST{i}", "false_positive", None)
        patterns = mgr.get_common_fp_patterns()  # no customer filter
        assert len(patterns) >= 1


# ==================================================================
# Threshold adjustment suggestions
# ==================================================================

class TestThresholdSuggestions:
    def test_suggests_tightening(self, mgr, cache):
        # 4 FP + 1 confirmed = 80% FP rate → should suggest tightening
        for i in range(4):
            cache.record_feedback(
                f"SA-dup-{i}", "CUST1", "false_positive", None)
        cache.record_feedback("SA-dup-cf", "CUST1", "confirmed_fraud", None)

        suggestions = mgr.suggest_threshold_adjustments("CUST1")
        dup_key = ModuleName.DUPLICATE_DETECTION.value
        assert dup_key in suggestions
        s = suggestions[dup_key]
        assert s["suggested_threshold"] > s["current_threshold"]
        assert s["fp_rate"] >= 0.50

    def test_no_suggestion_when_low_fp(self, mgr, cache):
        # 1 FP + 9 confirmed = 10% FP rate → no suggestion
        cache.record_feedback("SA-dup-fp", "CUST1", "false_positive", None)
        for i in range(9):
            cache.record_feedback(
                f"SA-dup-cf-{i}", "CUST1", "confirmed_fraud", None)

        suggestions = mgr.suggest_threshold_adjustments("CUST1")
        assert ModuleName.DUPLICATE_DETECTION.value not in suggestions

    def test_no_suggestion_insufficient_data(self, mgr, cache):
        # Only 2 reviews — below minimum of 5
        for i in range(2):
            cache.record_feedback(
                f"SA-dup-{i}", "CUST1", "false_positive", None)
        suggestions = mgr.suggest_threshold_adjustments("CUST1")
        assert ModuleName.DUPLICATE_DETECTION.value not in suggestions


# ==================================================================
# FP report
# ==================================================================

class TestFPReport:
    def test_report_structure(self, mgr, cache):
        mod = ModuleName.DUPLICATE_DETECTION.value
        for i in range(3):
            cache.record_feedback(
                f"SA-{mod}-fp{i}", "CUST1", "false_positive", None)
        cache.record_feedback(f"SA-{mod}-cf", "CUST1", "confirmed_fraud", None)

        report = mgr.generate_fp_report("CUST1")
        assert "overall_fp_rate" in report
        assert "per_module_rates" in report
        assert "common_patterns" in report
        assert "threshold_adjustments" in report
        assert report["customer_id"] == "CUST1"
        assert report["total_reviewed"] >= 4
        assert report["total_false_positives"] >= 3

    def test_empty_report(self, mgr):
        report = mgr.generate_fp_report("NOBODY")
        assert report["total_reviewed"] == 0
        assert report["overall_fp_rate"] == 0.0


# ==================================================================
# Edge cases
# ==================================================================

class TestEdgeCases:
    def test_empty_findings_list(self, mgr):
        result = mgr.apply_suppression_rules([], "CUST1")
        assert result == []

    def test_multiple_findings_independent(self, mgr, cache):
        dup_finding = _f(module=ModuleName.DUPLICATE_DETECTION,
                         evidence={"method": "exact_match"})
        price_finding = _f(module=ModuleName.PRICE_CREEP,
                           evidence={"method": "cumulative_drift"})

        dup_ph = _pattern_hash(dup_finding)
        for i in range(3):
            cache.record_feedback(
                f"SA-{dup_finding.module.value}-{dup_ph}-{i}",
                "CUST1", "false_positive", None)

        mgr.apply_suppression_rules([dup_finding, price_finding], "CUST1")
        assert dup_finding.suppressed is True
        assert price_finding.suppressed is False
