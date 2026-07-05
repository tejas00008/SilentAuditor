"""Tests for src.adjudication.conflict_resolver — ConflictResolver."""

from decimal import Decimal

import pytest

from src.adjudication.conflict_resolver import ConflictResolver, _escalate
from src.utils.constants import Finding, ModuleName, Severity, generate_finding_id


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _f(
    module: ModuleName = ModuleName.DUPLICATE_DETECTION,
    severity: Severity = Severity.REVIEW,
    confidence: float = 0.80,
    vendor_id: str = "V001",
    vendor_name: str = "Acme",
    invoice_ids: list[str] = None,
    amount: float = 5000.0,
    description: str = "Test finding",
    evidence: dict = None,
) -> Finding:
    """Shorthand factory for test findings."""
    return Finding(
        finding_id=generate_finding_id(),
        module=module,
        severity=severity,
        confidence=confidence,
        vendor_id=vendor_id,
        vendor_name=vendor_name,
        invoice_ids=invoice_ids or ["INV-001"],
        amount_at_risk=Decimal(str(amount)),
        description=description,
        evidence=evidence or {},
        recommended_action="Review",
    )


# ==================================================================
# _escalate helper
# ==================================================================

class TestEscalate:
    def test_info_to_review(self):
        assert _escalate(Severity.INFORMATIONAL) == Severity.REVIEW

    def test_review_to_critical(self):
        assert _escalate(Severity.REVIEW) == Severity.CRITICAL

    def test_critical_stays_critical(self):
        assert _escalate(Severity.CRITICAL) == Severity.CRITICAL


# ==================================================================
# Rule 1 — Duplicate vs Contract
# ==================================================================

class TestRule1DuplicateVsContract:
    def test_downgrade_when_contract_clean(self):
        """Contract exists for same invoice but no issues → downgrade dup."""
        dup = _f(
            module=ModuleName.DUPLICATE_DETECTION,
            confidence=0.90,
            invoice_ids=["INV-001", "INV-002"],
        )
        # Contract finding on same invoice that does NOT flag scope/rate
        contract = _f(
            module=ModuleName.CONTRACT_COMPLIANCE,
            invoice_ids=["INV-001"],
            description="Volume discount not applied",
            evidence={"check": "volume_commitment"},
        )
        resolver = ConflictResolver()
        resolver.resolve([dup, contract])
        assert dup.confidence == pytest.approx(0.90 * 0.6, abs=0.01)
        assert "milestone" in dup.description.lower()
        report = resolver.get_resolution_report()
        assert report["downgrades"] >= 1

    def test_no_downgrade_when_contract_flags_scope(self):
        """Contract also flags scope issue → keep duplicate at full confidence."""
        dup = _f(
            module=ModuleName.DUPLICATE_DETECTION,
            confidence=0.90,
            invoice_ids=["INV-001"],
        )
        contract = _f(
            module=ModuleName.CONTRACT_COMPLIANCE,
            invoice_ids=["INV-001"],
            description="Out-of-scope billing: marketing campaign",
            evidence={"check": "scope_boundary"},
        )
        original_conf = dup.confidence
        resolver = ConflictResolver()
        resolver.resolve([dup, contract])
        assert dup.confidence == original_conf

    def test_no_downgrade_without_contract(self):
        dup = _f(module=ModuleName.DUPLICATE_DETECTION, confidence=0.90)
        resolver = ConflictResolver()
        resolver.resolve([dup])
        assert dup.confidence == 0.90


# ==================================================================
# Rule 2 — Price Creep vs Market Price
# ==================================================================

class TestRule2PriceCreepVsMarket:
    def test_suppress_when_market_normal(self):
        """Market module found vendor but NOT in deviation → suppress creep."""
        creep = _f(
            module=ModuleName.PRICE_CREEP,
            vendor_id="V001",
        )
        market = _f(
            module=ModuleName.MARKET_PRICE,
            vendor_id="V001",
            evidence={"method": "renegotiation"},  # not deviation_scoring
        )
        resolver = ConflictResolver()
        resolver.resolve([creep, market])
        assert creep.suppressed is True
        assert "market trend" in creep.suppression_reason.lower()
        assert resolver.get_resolution_report()["suppressions"] >= 1

    def test_no_suppress_when_vendor_above_market(self):
        """Market module flags vendor as above-market → keep price creep."""
        creep = _f(module=ModuleName.PRICE_CREEP, vendor_id="V001")
        market = _f(
            module=ModuleName.MARKET_PRICE,
            vendor_id="V001",
            evidence={"method": "deviation_scoring"},
        )
        resolver = ConflictResolver()
        resolver.resolve([creep, market])
        assert creep.suppressed is False

    def test_no_suppress_without_market_findings(self):
        creep = _f(module=ModuleName.PRICE_CREEP, vendor_id="V001")
        resolver = ConflictResolver()
        resolver.resolve([creep])
        assert creep.suppressed is False


# ==================================================================
# Rule 3 — Split Invoicing vs Vendor Behavior
# ==================================================================

class TestRule3SplitVsBehavior:
    def test_suppress_when_behavior_normal(self):
        """Vendor behavior is only INFORMATIONAL → suppress split."""
        split = _f(
            module=ModuleName.SPLIT_INVOICING,
            vendor_id="V001",
        )
        behavior = _f(
            module=ModuleName.VENDOR_BEHAVIOR,
            vendor_id="V001",
            severity=Severity.INFORMATIONAL,
        )
        resolver = ConflictResolver()
        resolver.resolve([split, behavior])
        assert split.suppressed is True
        assert "baseline" in split.suppression_reason.lower()

    def test_no_suppress_when_behavior_anomalous(self):
        """Vendor behavior is REVIEW/CRITICAL → keep split."""
        split = _f(module=ModuleName.SPLIT_INVOICING, vendor_id="V001")
        behavior = _f(
            module=ModuleName.VENDOR_BEHAVIOR,
            vendor_id="V001",
            severity=Severity.REVIEW,
        )
        resolver = ConflictResolver()
        resolver.resolve([split, behavior])
        assert split.suppressed is False

    def test_no_suppress_without_behavior(self):
        split = _f(module=ModuleName.SPLIT_INVOICING, vendor_id="V001")
        resolver = ConflictResolver()
        resolver.resolve([split])
        assert split.suppressed is False


# ==================================================================
# Rule 4 — Phantom Services vs Contract
# ==================================================================

class TestRule4PhantomVsContract:
    def test_suppress_category_mismatch_when_contract_clean(self):
        """Contract has no scope issues → suppress phantom category mismatch."""
        phantom = _f(
            module=ModuleName.PHANTOM_SERVICES,
            vendor_id="V001",
            evidence={"method": "category_mismatch"},
        )
        contract = _f(
            module=ModuleName.CONTRACT_COMPLIANCE,
            vendor_id="V001",
            description="Rate overcharge",
            evidence={"check": "rate_compliance"},
        )
        resolver = ConflictResolver()
        resolver.resolve([phantom, contract])
        assert phantom.suppressed is True
        assert "scope" in phantom.suppression_reason.lower()

    def test_no_suppress_when_contract_flags_scope(self):
        """Contract also flags scope → keep phantom."""
        phantom = _f(
            module=ModuleName.PHANTOM_SERVICES,
            vendor_id="V001",
            evidence={"method": "category_mismatch"},
        )
        contract = _f(
            module=ModuleName.CONTRACT_COMPLIANCE,
            vendor_id="V001",
            evidence={"check": "scope_boundary"},
        )
        resolver = ConflictResolver()
        resolver.resolve([phantom, contract])
        assert phantom.suppressed is False

    def test_only_applies_to_category_and_oneoff(self):
        """Vagueness findings are NOT suppressed by this rule."""
        phantom = _f(
            module=ModuleName.PHANTOM_SERVICES,
            vendor_id="V001",
            evidence={"method": "vagueness"},
        )
        contract = _f(
            module=ModuleName.CONTRACT_COMPLIANCE,
            vendor_id="V001",
            evidence={"check": "rate_compliance"},
        )
        resolver = ConflictResolver()
        resolver.resolve([phantom, contract])
        assert phantom.suppressed is False


# ==================================================================
# Rule 5 — Multi-Module Corroboration
# ==================================================================

class TestRule5Corroboration:
    def test_two_modules_no_corroboration(self):
        """2 modules is below the 3-module threshold — no corroboration."""
        f1 = _f(
            module=ModuleName.DUPLICATE_DETECTION,
            severity=Severity.INFORMATIONAL,
            confidence=0.80,
            vendor_id="V001",
        )
        f2 = _f(
            module=ModuleName.PRICE_CREEP,
            severity=Severity.INFORMATIONAL,
            confidence=0.80,
            vendor_id="V001",
        )
        resolver = ConflictResolver()
        resolver.resolve([f1, f2])

        # 2 modules < 3 minimum → no corroboration, severity unchanged
        assert f1.severity == Severity.INFORMATIONAL
        assert f2.severity == Severity.INFORMATIONAL

    def test_three_modules_corroborated(self):
        """3+ qualified modules (conf>=0.75) → annotated with corroboration.

        Severity is NOT escalated (new behavior), but findings are linked.
        """
        f1 = _f(module=ModuleName.DUPLICATE_DETECTION,
                severity=Severity.INFORMATIONAL, confidence=0.80, vendor_id="V001")
        f2 = _f(module=ModuleName.PRICE_CREEP,
                severity=Severity.INFORMATIONAL, confidence=0.80, vendor_id="V001")
        f3 = _f(module=ModuleName.PHANTOM_SERVICES,
                severity=Severity.REVIEW, confidence=0.80, vendor_id="V001")
        resolver = ConflictResolver()
        resolver.resolve([f1, f2, f3])

        # Severity unchanged (new behavior — annotate only, don't escalate)
        assert f1.severity == Severity.INFORMATIONAL
        assert f2.severity == Severity.INFORMATIONAL
        assert f3.severity == Severity.REVIEW
        # But findings should be linked
        assert "Corroborated by 3 modules" in f1.description
        report = resolver.get_resolution_report()
        assert report["correlations"] >= 1

    def test_same_module_no_corroboration(self):
        """Two findings from the SAME module don't count as corroboration."""
        f1 = _f(module=ModuleName.DUPLICATE_DETECTION,
                severity=Severity.INFORMATIONAL, vendor_id="V001")
        f2 = _f(module=ModuleName.DUPLICATE_DETECTION,
                severity=Severity.INFORMATIONAL, vendor_id="V001")
        resolver = ConflictResolver()
        resolver.resolve([f1, f2])
        assert f1.severity == Severity.INFORMATIONAL

    def test_different_vendors_independent(self):
        """Different vendors are handled independently."""
        f1 = _f(module=ModuleName.DUPLICATE_DETECTION,
                severity=Severity.INFORMATIONAL, vendor_id="V001")
        f2 = _f(module=ModuleName.PRICE_CREEP,
                severity=Severity.INFORMATIONAL, vendor_id="V002")
        resolver = ConflictResolver()
        resolver.resolve([f1, f2])
        assert f1.severity == Severity.INFORMATIONAL
        assert f2.severity == Severity.INFORMATIONAL

    def test_suppressed_findings_excluded(self):
        """Suppressed findings don't participate in corroboration."""
        f1 = _f(module=ModuleName.DUPLICATE_DETECTION,
                severity=Severity.INFORMATIONAL, vendor_id="V001")
        f2 = _f(module=ModuleName.PRICE_CREEP,
                severity=Severity.INFORMATIONAL, vendor_id="V001")
        f2.suppressed = True
        f2.suppression_reason = "test"
        resolver = ConflictResolver()
        resolver.resolve([f1, f2])
        assert f1.severity == Severity.INFORMATIONAL


# ==================================================================
# Rule 6 — Collusion + Pricing
# ==================================================================

class TestRule6CollusionPricing:
    def test_escalate_cluster_with_pricing(self):
        """Collusion cluster + pricing anomaly → all to CRITICAL."""
        collusion = _f(
            module=ModuleName.VENDOR_COLLUSION,
            vendor_id="V001",
            severity=Severity.REVIEW,
            evidence={
                "method": "relationship_network",
                "vendor_a": "V001", "vendor_b": "V002",
                "cluster": ["V001", "V002"],
            },
        )
        pricing = _f(
            module=ModuleName.PRICE_CREEP,
            vendor_id="V001",
            severity=Severity.REVIEW,
        )
        # Another vendor in the cluster
        other = _f(
            module=ModuleName.PHANTOM_SERVICES,
            vendor_id="V002",
            severity=Severity.INFORMATIONAL,
        )
        resolver = ConflictResolver()
        resolver.resolve([collusion, pricing, other])

        assert pricing.severity == Severity.CRITICAL
        assert other.severity == Severity.CRITICAL
        assert "coordinated overpricing" in other.description.lower()

    def test_no_escalation_without_pricing(self):
        """Collusion without pricing anomaly → no rule-6 escalation."""
        collusion = _f(
            module=ModuleName.VENDOR_COLLUSION,
            vendor_id="V001",
            severity=Severity.REVIEW,
            evidence={
                "method": "relationship_network",
                "vendor_a": "V001", "vendor_b": "V002",
            },
        )
        other = _f(
            module=ModuleName.PHANTOM_SERVICES,
            vendor_id="V002",
            severity=Severity.INFORMATIONAL,
        )
        original_sev = other.severity
        resolver = ConflictResolver()
        resolver.resolve([collusion, other])
        # Rule 6 shouldn't fire; rule 5 might if same vendor
        # V001 and V002 are different → no corroboration either
        # (collusion is V001, other is V002 — different by_vendor groups)
        assert other.severity == original_sev


# ==================================================================
# Resolution report
# ==================================================================

class TestResolutionReport:
    def test_empty_findings(self):
        resolver = ConflictResolver()
        resolver.resolve([])
        report = resolver.get_resolution_report()
        assert report["total_resolutions"] == 0

    def test_report_counts(self):
        # Set up a scenario with a suppression and a corroboration
        creep = _f(module=ModuleName.PRICE_CREEP, vendor_id="V001")
        market = _f(
            module=ModuleName.MARKET_PRICE,
            vendor_id="V001",
            evidence={"method": "renegotiation"},
        )
        dup = _f(module=ModuleName.DUPLICATE_DETECTION, vendor_id="V002")
        phantom = _f(module=ModuleName.PHANTOM_SERVICES, vendor_id="V002")

        resolver = ConflictResolver()
        resolver.resolve([creep, market, dup, phantom])

        report = resolver.get_resolution_report()
        assert report["total_resolutions"] > 0
        assert isinstance(report["details"], list)


# ==================================================================
# Edge cases
# ==================================================================

class TestEdgeCases:
    def test_single_finding(self):
        f = _f()
        resolver = ConflictResolver()
        result = resolver.resolve([f])
        assert len(result) == 1
        assert f.suppressed is False

    def test_all_suppressed(self):
        """Already-suppressed findings are left alone."""
        f = _f()
        f.suppressed = True
        f.suppression_reason = "pre-existing"
        resolver = ConflictResolver()
        resolver.resolve([f])
        assert f.suppression_reason == "pre-existing"

    def test_returns_same_list(self):
        findings = [_f(), _f()]
        resolver = ConflictResolver()
        result = resolver.resolve(findings)
        assert result is findings
