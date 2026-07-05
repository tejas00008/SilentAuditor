"""Tests for src.detection.contract_compliance — ContractComplianceDetector."""

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.detection.contract_compliance import (
    ContractComplianceDetector,
    parse_payment_terms,
)
from src.normalization.cache import CacheManager
from src.utils.constants import (
    ContractRate,
    ContractTerms,
    ModuleName,
    Severity,
    VolumeDiscount,
)


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
        "rate_tolerance_pct": 3,
        "discount_window_buffer_days": 1,
        "scope_similarity_threshold": 0.70,
        "scope_ambiguous_lower": 0.50,
        "renewal_alert_days": 90,
        "volume_aggregation_period": "quarter",
    }


@pytest.fixture
def detector(config, cache, mock_llm):
    return ContractComplianceDetector(config, cache, mock_llm)


def _make_contract(**overrides) -> ContractTerms:
    defaults = dict(
        vendor_id="V001",
        contract_start_date=date(2024, 1, 1),
        contract_end_date=date(2025, 12, 31),
        auto_renewal=False,
        auto_renewal_notice_days=30,
        payment_terms="Net 30",
        rates=[ContractRate("Steel rebar", "ton", Decimal("100"))],
        volume_discounts=[VolumeDiscount(100, Decimal("5"))],
        scope_of_work=["Steel rebar delivery", "Concrete supply"],
        annual_escalation_pct=Decimal("3"),
    )
    defaults.update(overrides)
    return ContractTerms(**defaults)


def _make_inv_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "invoice_id": "", "invoice_number": "", "vendor_id": "V001",
        "vendor_name": "Acme", "invoice_date": "2025-01-15",
        "total_amount": 5000.0, "unit_price": 100.0,
        "quantity": 50, "line_item_description": "Steel rebar delivery",
        "payment_date": "2025-02-10",
    }
    full = []
    for r in rows:
        row = {**defaults, **r}
        if not row["invoice_id"]:
            row["invoice_id"] = row["invoice_number"]
        full.append(row)
    return pd.DataFrame(full)


# ==================================================================
# Payment-term parser
# ==================================================================

class TestParsePaymentTerms:
    def test_net_30(self):
        r = parse_payment_terms("Net 30")
        assert r == {"net_days": 30, "discount_pct": 0, "discount_days": 0}

    def test_2_10_net_30(self):
        r = parse_payment_terms("2/10 Net 30")
        assert r == {"net_days": 30, "discount_pct": 2, "discount_days": 10}

    def test_1_15_net_45(self):
        r = parse_payment_terms("1/15 Net 45")
        assert r == {"net_days": 45, "discount_pct": 1, "discount_days": 15}

    def test_net_60(self):
        r = parse_payment_terms("Net 60")
        assert r["net_days"] == 60

    def test_due_on_receipt(self):
        r = parse_payment_terms("Due on Receipt")
        assert r["net_days"] == 0

    def test_cod(self):
        r = parse_payment_terms("COD")
        assert r["net_days"] == 0

    def test_empty(self):
        r = parse_payment_terms("")
        assert r["net_days"] == 0


# ==================================================================
# Module identity
# ==================================================================

class TestModuleIdentity:
    def test_module_name(self, detector):
        assert detector.get_module_name() == ModuleName.CONTRACT_COMPLIANCE

    def test_required_fields(self, detector):
        assert "unit_price" in detector.get_required_fields()

    def test_no_contracts_no_findings(self, detector):
        df = _make_inv_df([{"invoice_number": "I1"}])
        findings = detector.detect(df, supplementary_data={})
        assert findings == []


# ==================================================================
# Check 1 — Rate Compliance
# ==================================================================

class TestRateCompliance:
    def test_flags_overcharge(self, detector):
        contract = _make_contract(
            rates=[ContractRate("Steel rebar", "ton", Decimal("100"))],
            annual_escalation_pct=Decimal("3"),
        )
        df = _make_inv_df([{
            "invoice_number": "I1", "unit_price": 120,
            "line_item_description": "Steel rebar grade 60",
        }])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        rate_f = [f for f in findings if f.evidence.get("check") == "rate_compliance"]
        assert len(rate_f) >= 1
        assert rate_f[0].evidence["deviation_pct"] > 10

    def test_no_flag_within_tolerance(self, detector):
        contract = _make_contract(
            rates=[ContractRate("Steel rebar", "ton", Decimal("100"))],
            annual_escalation_pct=Decimal("3"),
        )
        df = _make_inv_df([{
            "invoice_number": "I1", "unit_price": 103,
            "line_item_description": "Steel rebar delivery",
        }])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        rate_f = [f for f in findings if f.evidence.get("check") == "rate_compliance"]
        assert len(rate_f) == 0

    def test_accounts_for_escalation(self, detector):
        contract = _make_contract(
            rates=[ContractRate("Steel rebar", "ton", Decimal("100"))],
            annual_escalation_pct=Decimal("3"),
            contract_start_date=date(2023, 1, 1),
        )
        # 2 years later: allowed = 100 * 1.03^2 ≈ 106.09
        df = _make_inv_df([{
            "invoice_number": "I1", "unit_price": 107,
            "invoice_date": "2025-01-15",
            "line_item_description": "Steel rebar delivery",
        }])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        rate_f = [f for f in findings if f.evidence.get("check") == "rate_compliance"]
        # 107 is within ~103% of 106.09 → within tolerance
        assert len(rate_f) == 0


# ==================================================================
# Check 2 — Payment Term Compliance
# ==================================================================

class TestPaymentTerms:
    def test_flags_missed_discount(self, detector):
        contract = _make_contract(payment_terms="2/10 Net 30")
        df = _make_inv_df([{
            "invoice_number": "I1", "total_amount": 10000,
            "invoice_date": date(2025, 1, 1),
            "payment_date": date(2025, 1, 8),  # within 10 days
            "line_item_description": "Steel rebar",
        }])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        pay_f = [f for f in findings if f.evidence.get("check") == "payment_terms"]
        assert len(pay_f) >= 1
        assert pay_f[0].evidence["expected_discount"] == 200.0

    def test_no_flag_without_discount_terms(self, detector):
        contract = _make_contract(payment_terms="Net 30")
        df = _make_inv_df([{
            "invoice_number": "I1", "total_amount": 10000,
            "invoice_date": date(2025, 1, 1),
            "payment_date": date(2025, 1, 5),
            "line_item_description": "Steel rebar",
        }])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        pay_f = [f for f in findings if f.evidence.get("check") == "payment_terms"]
        assert len(pay_f) == 0

    def test_no_flag_paid_after_discount_window(self, detector):
        contract = _make_contract(payment_terms="2/10 Net 30")
        df = _make_inv_df([{
            "invoice_number": "I1", "total_amount": 10000,
            "invoice_date": date(2025, 1, 1),
            "payment_date": date(2025, 1, 25),  # after 10-day window
            "line_item_description": "Steel rebar",
        }])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        pay_f = [f for f in findings if f.evidence.get("check") == "payment_terms"]
        assert len(pay_f) == 0


# ==================================================================
# Check 3 — Volume Commitment
# ==================================================================

class TestVolumeCommitment:
    def test_flags_unapplied_discount(self, detector):
        contract = _make_contract(
            volume_discounts=[VolumeDiscount(100, Decimal("5"))],
        )
        rows = [
            {"invoice_number": f"I{i}", "quantity": 25,
             "total_amount": 2500, "unit_price": 100,
             "line_item_description": "Steel rebar"}
            for i in range(5)
        ]  # total qty = 125 > threshold 100
        df = _make_inv_df(rows)
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        vol_f = [f for f in findings if f.evidence.get("check") == "volume_commitment"]
        assert len(vol_f) >= 1
        assert vol_f[0].evidence["total_quantity"] == 125

    def test_no_flag_below_threshold(self, detector):
        contract = _make_contract(
            volume_discounts=[VolumeDiscount(100, Decimal("5"))],
        )
        rows = [
            {"invoice_number": f"I{i}", "quantity": 10,
             "total_amount": 1000, "unit_price": 100,
             "line_item_description": "Steel rebar"}
            for i in range(3)
        ]  # total qty = 30 < threshold 100
        df = _make_inv_df(rows)
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        vol_f = [f for f in findings if f.evidence.get("check") == "volume_commitment"]
        assert len(vol_f) == 0


# ==================================================================
# Check 4 — Scope Boundary
# ==================================================================

class TestScopeBoundary:
    def test_flags_out_of_scope(self, detector):
        contract = _make_contract(
            scope_of_work=["Steel rebar delivery", "Concrete supply"],
        )
        df = _make_inv_df([{
            "invoice_number": "I1", "total_amount": 5000,
            "line_item_description": "Strategic marketing campaign development",
        }])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        scope_f = [f for f in findings if f.evidence.get("check") == "scope_boundary"]
        assert len(scope_f) >= 1

    def test_no_flag_for_in_scope(self, detector):
        contract = _make_contract(
            scope_of_work=["Steel rebar delivery", "Concrete supply"],
        )
        df = _make_inv_df([{
            "invoice_number": "I1", "total_amount": 5000,
            "line_item_description": "Steel rebar delivery grade 60",
        }])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        scope_f = [f for f in findings if f.evidence.get("check") == "scope_boundary"]
        assert len(scope_f) == 0

    def test_small_amount_skipped(self, detector):
        contract = _make_contract(scope_of_work=["Plumbing"])
        df = _make_inv_df([{
            "invoice_number": "I1", "total_amount": 100,
            "line_item_description": "Random unrelated stuff xyz",
        }])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        scope_f = [f for f in findings if f.evidence.get("check") == "scope_boundary"]
        assert len(scope_f) == 0


# ==================================================================
# Check 5 — Contract Expiration
# ==================================================================

class TestContractExpiration:
    def test_flags_post_expiry(self, detector):
        # Contract expired 2024-09-30, invoice 2025-01-15 = 107 days gap
        # New behavior: consolidated finding, REVIEW for long gap (>90 days)
        contract = _make_contract(contract_end_date=date(2024, 9, 30))
        df = _make_inv_df([{
            "invoice_number": "I1",
            "invoice_date": date(2025, 1, 15),
            "line_item_description": "Steel rebar",
        }])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        exp_f = [f for f in findings if f.evidence.get("check") == "expiration"]
        assert len(exp_f) >= 1
        assert exp_f[0].severity in (Severity.REVIEW, Severity.CRITICAL)
        assert exp_f[0].evidence["max_days_past_expiry"] > 90

    def test_no_flag_before_expiry(self, detector):
        contract = _make_contract(contract_end_date=date(2026, 12, 31))
        df = _make_inv_df([{
            "invoice_number": "I1",
            "invoice_date": date(2025, 6, 1),
            "line_item_description": "Steel rebar",
        }])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        exp_f = [f for f in findings if f.evidence.get("check") == "expiration"]
        assert len(exp_f) == 0


# ==================================================================
# Check 6 — Escalation Clause
# ==================================================================

class TestEscalationClause:
    def test_flags_excess_escalation(self, detector):
        contract = _make_contract(
            rates=[ContractRate("Steel rebar", "ton", Decimal("100"))],
            annual_escalation_pct=Decimal("3"),
            contract_start_date=date(2024, 1, 1),
        )
        # Price is 110 → 10% escalation vs max 3%
        df = _make_inv_df([{
            "invoice_number": "I1", "unit_price": 110,
            "invoice_date": date(2025, 3, 1),
            "line_item_description": "Steel rebar delivery",
            "quantity": 20,
        }])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        esc_f = [f for f in findings if f.evidence.get("check") == "escalation"]
        assert len(esc_f) >= 1
        assert esc_f[0].evidence["actual_escalation_pct"] > 3

    def test_no_flag_within_limit(self, detector):
        contract = _make_contract(
            rates=[ContractRate("Steel rebar", "ton", Decimal("100"))],
            annual_escalation_pct=Decimal("5"),
        )
        df = _make_inv_df([{
            "invoice_number": "I1", "unit_price": 103,
            "line_item_description": "Steel rebar delivery",
        }])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        esc_f = [f for f in findings if f.evidence.get("check") == "escalation"]
        assert len(esc_f) == 0


# ==================================================================
# Edge cases
# ==================================================================

class TestEdgeCases:
    def test_empty_dataframe(self, detector):
        contract = _make_contract()
        df = pd.DataFrame(columns=[
            "invoice_id", "vendor_id", "total_amount", "invoice_date",
            "line_item_description", "unit_price",
        ])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        assert findings == []

    def test_no_matching_vendor(self, detector):
        contract = _make_contract(vendor_id="V999")
        df = _make_inv_df([{"invoice_number": "I1", "vendor_id": "V001"}])
        findings = detector.detect(df, supplementary_data={"contracts": [contract]})
        assert len(findings) == 0

    def test_stats_populated(self, detector):
        contract = _make_contract()
        df = _make_inv_df([{"invoice_number": "I1"}])
        detector.detect(df, supplementary_data={"contracts": [contract]})
        stats = detector.get_stats()
        assert stats["module"] == "contract_compliance"


# ==================================================================
# Sample data integration
# ==================================================================

class TestSampleDataIntegration:
    @pytest.fixture
    def sample_data(self):
        import json as _json
        from pathlib import Path

        base = Path(__file__).resolve().parent.parent / "data"
        inv_path = base / "sample_invoices" / "invoices.csv"
        key_path = base / "sample_invoices" / "fraud_key.json"
        con_path = base / "sample_contracts" / "contracts.json"

        if not inv_path.exists():
            pytest.skip("Sample data not generated")

        df = pd.read_csv(inv_path, dtype=str, keep_default_na=False)
        for col in ["total_amount", "unit_price", "quantity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "invoice_id" not in df.columns:
            df["invoice_id"] = df["invoice_number"]

        with open(key_path) as f:
            fraud_key = _json.load(f)
        with open(con_path) as f:
            contracts = _json.load(f)

        return df, fraud_key, contracts

    def test_detects_contract_violations(self, config, cache, mock_llm, sample_data):
        df, fraud_key, contracts = sample_data
        det = ContractComplianceDetector(config, cache, mock_llm)
        findings = det.detect(df, supplementary_data={"contracts": contracts})

        cc_frauds = [
            f for f in fraud_key["fraud_items"]
            if f["fraud_type"] == "contract_compliance"
        ]
        assert len(cc_frauds) == 10

        detected = 0
        for fraud in cc_frauds:
            fraud_ids = set(fraud.get("invoice_ids", []))
            fraud_vendor = fraud.get("vendor", "")
            for finding in findings:
                if not finding.suppressed:
                    fids = set(finding.invoice_ids)
                    if fraud_ids & fids or finding.vendor_name in fraud_vendor:
                        detected += 1
                        break

        assert detected >= 3, (
            f"Detected {detected}/10 contract violations"
        )

    def test_multiple_checks_fire(self, config, cache, mock_llm, sample_data):
        df, _, contracts = sample_data
        det = ContractComplianceDetector(config, cache, mock_llm)
        findings = det.detect(df, supplementary_data={"contracts": contracts})
        checks = {f.evidence.get("check") for f in findings}
        assert len(checks) >= 2, f"Expected >=2 check types, got: {checks}"
