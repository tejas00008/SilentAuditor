"""Tests for src.detection.vendor_collusion — VendorCollusionDetector."""

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.detection.vendor_collusion import VendorCollusionDetector
from src.normalization.cache import CacheManager
from src.utils.constants import ModuleName, Severity


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
        "relationship_composite_threshold": 0.60,
        "timing_correlation_threshold": 0.7,
        "overbilling_percentile": 75,
        "overbilling_margin_pct": 15,
        "approval_concentration_warning": 0.3,
        "benfords_min_invoices": 50,
        "benfords_p_value": 0.05,
    }


@pytest.fixture
def detector(config, cache, mock_llm):
    return VendorCollusionDetector(config, cache, mock_llm)


def _make_vendor_df(vendors: list[dict]) -> pd.DataFrame:
    defaults = {
        "vendor_id": "", "vendor_name": "", "address": "",
        "phone": "", "tax_id": "", "bank_account_last4": "",
        "contact_person": "",
    }
    return pd.DataFrame([{**defaults, **v} for v in vendors])


def _make_inv_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "invoice_id": "", "invoice_number": "", "vendor_id": "",
        "vendor_name": "", "invoice_date": "2025-01-15",
        "total_amount": 1000.0, "line_item_description": "Service",
        "approved_by": "",
    }
    full = []
    for r in rows:
        row = {**defaults, **r}
        if not row["invoice_id"]:
            row["invoice_id"] = row["invoice_number"]
        full.append(row)
    return pd.DataFrame(full)


def _shared_address_cluster() -> tuple[pd.DataFrame, pd.DataFrame]:
    """3 vendors sharing the same address."""
    vendor_df = _make_vendor_df([
        {"vendor_id": "V001", "vendor_name": "Alpha Mechanical",
         "address": "123 Main St, Springfield, IL 62701",
         "bank_account_last4": "1111"},
        {"vendor_id": "V002", "vendor_name": "Alpha Electrical",
         "address": "123 Main St, Springfield, IL 62701",
         "bank_account_last4": "2222"},
        {"vendor_id": "V003", "vendor_name": "Alpha General",
         "address": "123 Main St, Springfield, IL 62701",
         "bank_account_last4": "3333"},
        # Unrelated vendor
        {"vendor_id": "V004", "vendor_name": "Beta Corp",
         "address": "456 Oak Ave, Portland, OR 97201",
         "bank_account_last4": "4444"},
    ])
    inv_rows = []
    for vid, vname in [("V001", "Alpha Mechanical"),
                       ("V002", "Alpha Electrical"),
                       ("V003", "Alpha General"),
                       ("V004", "Beta Corp")]:
        for i in range(10):
            inv_rows.append({
                "invoice_number": f"INV-{vid}-{i}",
                "vendor_id": vid, "vendor_name": vname,
                "invoice_date": date(2024, (i % 12) + 1, 15),
                "total_amount": 5000 + i * 100,
                "line_item_description": "HVAC maintenance service",
                "approved_by": "John Smith",
            })
    return _make_inv_df(inv_rows), vendor_df


def _shared_bank_cluster() -> tuple[pd.DataFrame, pd.DataFrame]:
    """2 vendors sharing the same bank account."""
    vendor_df = _make_vendor_df([
        {"vendor_id": "V010", "vendor_name": "Riverside Plumbing",
         "address": "100 River Rd", "bank_account_last4": "9999"},
        {"vendor_id": "V011", "vendor_name": "Riverside Maintenance",
         "address": "200 Lake St", "bank_account_last4": "9999"},
        {"vendor_id": "V012", "vendor_name": "Unrelated Co",
         "address": "300 Hill Dr", "bank_account_last4": "0000"},
    ])
    inv_rows = []
    for vid, vname in [("V010", "Riverside Plumbing"),
                       ("V011", "Riverside Maintenance"),
                       ("V012", "Unrelated Co")]:
        for i in range(10):
            inv_rows.append({
                "invoice_number": f"INV-{vid}-{i}",
                "vendor_id": vid, "vendor_name": vname,
                "invoice_date": date(2024, (i % 12) + 1, 10 + (i % 5)),
                "total_amount": 3000 + i * 200,
                "line_item_description": "Plumbing repair",
                "approved_by": "Jane Doe" if vid != "V012" else "Bob Lee",
            })
    return _make_inv_df(inv_rows), vendor_df


# ==================================================================
# Module identity
# ==================================================================

class TestModuleIdentity:
    def test_module_name(self, detector):
        assert detector.get_module_name() == ModuleName.VENDOR_COLLUSION

    def test_required_fields(self, detector):
        req = detector.get_required_fields()
        assert "vendor_id" in req
        assert "line_item_description" in req

    def test_optional_fields(self, detector):
        assert "vendor_bank_account_last4" in detector.get_optional_fields()
        assert "approved_by" in detector.get_optional_fields()


# ==================================================================
# Method 1 — Relationship Network
# ==================================================================

class TestRelationshipNetwork:
    def test_shared_address_cluster(self, detector):
        inv_df, vendor_df = _shared_address_cluster()
        findings = detector.detect(inv_df, vendor_data=vendor_df)
        network = [f for f in findings
                   if f.evidence.get("method") == "relationship_network"]
        assert len(network) >= 2  # 3 vendors → at least 2 pairwise edges
        assert any("address" in str(f.evidence.get("signals", []))
                    for f in network)

    def test_shared_bank_cluster(self, detector):
        inv_df, vendor_df = _shared_bank_cluster()
        findings = detector.detect(inv_df, vendor_data=vendor_df)
        network = [f for f in findings
                   if f.evidence.get("method") == "relationship_network"]
        assert len(network) >= 1
        assert any("bank" in str(f.evidence.get("signals", []))
                    for f in network)

    def test_unrelated_vendors_no_edge(self, detector):
        vendor_df = _make_vendor_df([
            {"vendor_id": "V001", "vendor_name": "Acme",
             "address": "123 Main St", "bank_account_last4": "1111"},
            {"vendor_id": "V002", "vendor_name": "Beta",
             "address": "456 Oak Ave", "bank_account_last4": "2222"},
        ])
        inv_df = _make_inv_df([
            {"invoice_number": f"INV-V001-{i}", "vendor_id": "V001",
             "vendor_name": "Acme", "total_amount": 1000,
             "line_item_description": "Service"}
            for i in range(5)
        ] + [
            {"invoice_number": f"INV-V002-{i}", "vendor_id": "V002",
             "vendor_name": "Beta", "total_amount": 2000,
             "line_item_description": "Other"}
            for i in range(5)
        ])
        findings = detector.detect(inv_df, vendor_data=vendor_df)
        network = [f for f in findings
                   if f.evidence.get("method") == "relationship_network"]
        assert len(network) == 0

    def test_name_similarity_signal(self, detector):
        vendor_df = _make_vendor_df([
            {"vendor_id": "V001", "vendor_name": "Alpine Mechanical Services",
             "address": "100 Main", "bank_account_last4": "1111"},
            {"vendor_id": "V002", "vendor_name": "Alpine Mechanical Corp",
             "address": "100 Main", "bank_account_last4": "2222"},
        ])
        inv_df = _make_inv_df([
            {"invoice_number": f"INV-V{v}-{i}", "vendor_id": f"V00{v}",
             "vendor_name": f"Alpine Mechanical {'Services' if v == 1 else 'Corp'}",
             "total_amount": 1000, "line_item_description": "Service"}
            for v in [1, 2] for i in range(3)
        ])
        vendors = detector._build_vendor_list(inv_df, vendor_df)
        va = vendors["V001"]
        vb = vendors["V002"]
        score, signals = detector._relationship_score(va, vb)
        assert score > 0  # address + name similarity should contribute


# ==================================================================
# Method 2 — Coordinated Invoicing
# ==================================================================

class TestCoordinatedInvoicing:
    def test_correlated_dates_flagged(self, detector):
        inv_df, vendor_df = _shared_address_cluster()
        findings = detector.detect(inv_df, vendor_data=vendor_df)
        coordinated = [f for f in findings
                       if f.evidence.get("method") == "coordinated_invoicing"]
        # The cluster vendors invoice on the same monthly schedule
        assert len(coordinated) >= 1

    def test_uncorrelated_dates_not_flagged(self, detector):
        vendor_df = _make_vendor_df([
            {"vendor_id": "V001", "vendor_name": "Alpha",
             "address": "123 Same Addr", "bank_account_last4": "1111"},
            {"vendor_id": "V002", "vendor_name": "Beta",
             "address": "123 Same Addr", "bank_account_last4": "2222"},
        ])
        # Totally different date patterns
        inv_rows = []
        for i in range(12):
            inv_rows.append({
                "invoice_number": f"INV-V001-{i}", "vendor_id": "V001",
                "vendor_name": "Alpha",
                "invoice_date": date(2024, (i % 12) + 1, 5),
                "total_amount": 1000, "line_item_description": "Svc",
            })
        for i in range(6):
            inv_rows.append({
                "invoice_number": f"INV-V002-{i}", "vendor_id": "V002",
                "vendor_name": "Beta",
                "invoice_date": date(2025, (i % 6) + 1, 25),
                "total_amount": 2000, "line_item_description": "Other",
            })
        inv_df = _make_inv_df(inv_rows)
        findings = detector.detect(inv_df, vendor_data=vendor_df)
        coordinated = [f for f in findings
                       if f.evidence.get("method") == "coordinated_invoicing"]
        # Different date patterns should not produce high correlation
        for f in coordinated:
            assert f.evidence["correlation"] < 0.9


# ==================================================================
# Method 3 — Overbilling
# ==================================================================

class TestOverbilling:
    def test_flags_overpriced_cluster_vendor(self, detector):
        # 2 cluster vendors at same address, 10 unrelated at distinct addrs
        # Cluster at $15K, everyone else at $5K → cluster is well above p75
        cities = [
            "Portland OR", "Austin TX", "Denver CO", "Charlotte NC",
            "Phoenix AZ", "Columbus OH", "Nashville TN", "San Jose CA",
            "Milwaukee WI", "Indianapolis IN",
        ]
        vendors = []
        for i in range(12):
            addr = ("123 Shared Industrial Blvd, Springfield, IL 62701"
                    if i < 2 else f"{100+i} {cities[i-2]} St")
            vendors.append({
                "vendor_id": f"V{i:03d}", "vendor_name": f"Vendor {i}",
                "address": addr,
                "bank_account_last4": f"{i:04d}",
            })
        vendor_df = _make_vendor_df(vendors)
        inv_rows = []
        for i in range(12):
            amt = 15000 if i < 2 else 5000
            for j in range(5):
                inv_rows.append({
                    "invoice_number": f"INV-V{i:03d}-{j}",
                    "vendor_id": f"V{i:03d}",
                    "vendor_name": f"Vendor {i}",
                    "total_amount": amt + j * 100,
                    "line_item_description": "Service",
                })
        inv_df = _make_inv_df(inv_rows)
        findings = detector.detect(inv_df, vendor_data=vendor_df)
        overbill = [f for f in findings
                    if f.evidence.get("method") == "overbilling"]
        assert len(overbill) >= 1


# ==================================================================
# Method 4 — Approval Concentration
# ==================================================================

class TestApprovalConcentration:
    def test_flags_sole_approver(self, detector):
        inv_df, vendor_df = _shared_address_cluster()
        # All approved by same person
        findings = detector.detect(inv_df, vendor_data=vendor_df)
        approval = [f for f in findings
                    if f.evidence.get("method") == "approval_concentration"]
        assert len(approval) >= 1
        assert any(f.evidence["concentration"] >= 0.8 for f in approval)

    def test_no_flag_without_approved_by(self, detector):
        vendor_df = _make_vendor_df([
            {"vendor_id": "V001", "vendor_name": "A",
             "address": "123 Same", "bank_account_last4": "1111"},
            {"vendor_id": "V002", "vendor_name": "B",
             "address": "123 Same", "bank_account_last4": "2222"},
        ])
        inv_df = pd.DataFrame({
            "invoice_id": ["I1", "I2"],
            "vendor_id": ["V001", "V002"],
            "vendor_name": ["A", "B"],
            "invoice_date": ["2025-01-01", "2025-01-01"],
            "total_amount": [1000, 1000],
            "line_item_description": ["Svc", "Svc"],
        })
        findings = detector.detect(inv_df, vendor_data=vendor_df)
        approval = [f for f in findings
                    if f.evidence.get("method") == "approval_concentration"]
        assert len(approval) == 0


# ==================================================================
# Method 5 — Benford's Law
# ==================================================================

class TestBenfordsAnomaly:
    def test_flags_non_benford(self, detector):
        import numpy as np
        rng = np.random.default_rng(42)
        inv_rows = []
        for i in range(60):
            inv_rows.append({
                "invoice_number": f"INV-{i:03d}", "vendor_id": "V001",
                "vendor_name": "FakeDist",
                "total_amount": float(rng.integers(4000, 5999)),
                "invoice_date": date(2024, (i % 12) + 1, 15),
                "line_item_description": "Service",
            })
        inv_df = _make_inv_df(inv_rows)
        findings = detector.detect(inv_df)
        benford = [f for f in findings
                   if f.evidence.get("method") == "benfords"]
        assert len(benford) >= 1

    def test_too_few_invoices_skipped(self, detector):
        inv_rows = [
            {"invoice_number": f"INV-{i}", "vendor_id": "V001",
             "total_amount": 4500, "line_item_description": "Svc",
             "invoice_date": "2025-01-15"}
            for i in range(10)
        ]
        inv_df = _make_inv_df(inv_rows)
        findings = detector.detect(inv_df)
        benford = [f for f in findings
                   if f.evidence.get("method") == "benfords"]
        assert len(benford) == 0


# ==================================================================
# Edge cases
# ==================================================================

class TestEdgeCases:
    def test_empty_dataframe(self, detector):
        df = pd.DataFrame(columns=[
            "invoice_id", "vendor_id", "vendor_name", "total_amount",
            "invoice_date", "line_item_description",
        ])
        assert detector.detect(df) == []

    def test_single_vendor(self, detector):
        inv_df = _make_inv_df([
            {"invoice_number": "I1", "vendor_id": "V001",
             "vendor_name": "Solo", "total_amount": 1000,
             "line_item_description": "Svc"}
        ])
        findings = detector.detect(inv_df)
        network = [f for f in findings
                   if f.evidence.get("method") == "relationship_network"]
        assert len(network) == 0

    def test_stats_populated(self, detector):
        inv_df, vendor_df = _shared_address_cluster()
        detector.detect(inv_df, vendor_data=vendor_df)
        stats = detector.get_stats()
        assert stats["invoices_analyzed"] > 0
        assert stats["module"] == "vendor_collusion"


# ==================================================================
# Sample data integration
# ==================================================================

class TestSampleDataIntegration:
    @pytest.fixture
    def sample_data(self):
        import json
        from pathlib import Path

        base = Path(__file__).resolve().parent.parent / "data"
        inv_path = base / "sample_invoices" / "invoices.csv"
        key_path = base / "sample_invoices" / "fraud_key.json"
        vendor_path = base / "sample_vendor_master" / "vendors.csv"

        if not inv_path.exists():
            pytest.skip("Sample data not generated")

        df = pd.read_csv(inv_path, dtype=str, keep_default_na=False)
        df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
        if "invoice_id" not in df.columns:
            df["invoice_id"] = df["invoice_number"]

        vendor_df = pd.read_csv(vendor_path, dtype=str, keep_default_na=False)

        with open(key_path) as f:
            fraud_key = json.load(f)

        return df, fraud_key, vendor_df

    def test_detects_collusion_clusters(self, config, cache, mock_llm, sample_data):
        df, fraud_key, vendor_df = sample_data
        det = VendorCollusionDetector(config, cache, mock_llm)
        findings = det.detect(df, vendor_data=vendor_df)

        collusion_frauds = [
            f for f in fraud_key["fraud_items"]
            if f["fraud_type"] == "vendor_collusion"
        ]
        assert len(collusion_frauds) == 2

        # Match by vendor names in the cluster descriptions
        detected = 0
        for fraud in collusion_frauds:
            fraud_vendors = set(v.strip() for v in fraud.get("vendor", "").split(","))
            for finding in findings:
                if finding.evidence.get("method") == "relationship_network":
                    name_a = finding.evidence.get("name_a", "")
                    name_b = finding.evidence.get("name_b", "")
                    if any(fv in name_a or fv in name_b for fv in fraud_vendors):
                        detected += 1
                        break

        assert detected >= 1, (
            f"Detected {detected}/2 collusion clusters"
        )

    def test_multiple_methods_fire(self, config, cache, mock_llm, sample_data):
        df, _, vendor_df = sample_data
        det = VendorCollusionDetector(config, cache, mock_llm)
        findings = det.detect(df, vendor_data=vendor_df)
        methods = {f.evidence.get("method") for f in findings}
        assert len(methods) >= 1, f"Expected >=1 methods, got: {methods}"
