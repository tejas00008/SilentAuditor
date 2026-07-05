"""Tests for src.detection.split_invoicing — SplitInvoiceDetector."""

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.detection.split_invoicing import SplitInvoiceDetector
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
        "temporal_windows_days": [7, 14, 30],
        "min_invoices_in_cluster": 3,
        "below_threshold_band_pct": 10,
        "line_item_similarity_threshold": 0.75,
        "benfords_mad_first_digit": 0.015,
        "benfords_mad_first_two": 0.012,
        "benfords_min_sample": 50,
        "historical_amount_drop_pct": 30,
        "default_approval_thresholds": [5000, 25000, 100000],
    }


@pytest.fixture
def detector(config, cache, mock_llm):
    return SplitInvoiceDetector(config, cache, mock_llm)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "invoice_id": "", "invoice_number": "", "vendor_id": "",
        "vendor_name": "", "invoice_date": "2025-01-15",
        "total_amount": 1000.0, "po_number": "",
        "project_code": "", "line_item_description": "",
    }
    full = []
    for r in rows:
        row = {**defaults, **r}
        if not row["invoice_id"]:
            row["invoice_id"] = row["invoice_number"]
        full.append(row)
    return pd.DataFrame(full)


# ==================================================================
# Module identity
# ==================================================================

class TestModuleIdentity:
    def test_module_name(self, detector):
        assert detector.get_module_name() == ModuleName.SPLIT_INVOICING

    def test_required_fields(self, detector):
        req = detector.get_required_fields()
        assert "vendor_id" in req
        assert "total_amount" in req
        assert "invoice_date" in req

    def test_optional_fields(self, detector):
        opt = detector.get_optional_fields()
        assert "po_number" in opt


# ==================================================================
# Threshold loading
# ==================================================================

class TestGetThresholds:
    def test_from_supplementary_data(self, detector):
        supp = {"approval_thresholds": [
            {"level": "Mgr", "max_amount": 3000},
            {"level": "Dir", "max_amount": 10000},
        ]}
        result = detector._get_thresholds(supp)
        assert result == [Decimal("3000"), Decimal("10000")]

    def test_defaults_from_config(self, detector):
        result = detector._get_thresholds({})
        assert result == [Decimal("5000"), Decimal("25000"), Decimal("100000")]

    def test_skips_none_amounts(self, detector):
        supp = {"approval_thresholds": [
            {"level": "Mgr", "max_amount": 5000},
            {"level": "Board", "max_amount": None},
        ]}
        result = detector._get_thresholds(supp)
        assert result == [Decimal("5000")]


# ==================================================================
# Method 1: Threshold Clustering
# ==================================================================

class TestThresholdClustering:
    def test_detects_clustering_below_5k(self, detector):
        # 30 invoices spread across low range, plus 10 clustering just below $5000
        # The heavy cluster at the top must be statistically significant (p<0.01)
        rows = []
        for i in range(30):
            rows.append({
                "invoice_number": f"INV-{i:03d}", "vendor_id": "V001",
                "vendor_name": "Acme", "total_amount": 500 + i * 100,
                "invoice_date": date(2024, (i % 12) + 1, 15),
            })
        # 10 invoices clustering in $4800-$4999
        for i in range(10):
            rows.append({
                "invoice_number": f"INV-C{i:03d}", "vendor_id": "V001",
                "vendor_name": "Acme", "total_amount": 4800 + i * 20,
                "invoice_date": date(2025, (i % 6) + 1, 15),
            })
        df = _make_df(rows)
        findings = detector.detect(df)
        clustering = [f for f in findings
                      if f.evidence.get("method") == "threshold_clustering"]
        assert len(clustering) >= 1
        assert clustering[0].evidence["threshold"] == 5000.0

    def test_no_clustering_uniform(self, detector):
        # 25 invoices uniformly distributed — no clustering
        rows = []
        for i in range(25):
            rows.append({
                "invoice_number": f"INV-{i:03d}", "vendor_id": "V001",
                "vendor_name": "Acme",
                "total_amount": 200 + i * 180,  # 200 to 4520
                "invoice_date": date(2024, (i % 12) + 1, 15),
            })
        df = _make_df(rows)
        findings = detector.detect(df)
        clustering = [f for f in findings
                      if f.evidence.get("method") == "threshold_clustering"]
        assert len(clustering) == 0

    def test_too_few_invoices_skipped(self, detector):
        rows = [
            {"invoice_number": f"INV-{i}", "vendor_id": "V001",
             "total_amount": 4900, "invoice_date": "2025-01-15"}
            for i in range(5)
        ]
        df = _make_df(rows)
        findings = detector.detect(df)
        clustering = [f for f in findings
                      if f.evidence.get("method") == "threshold_clustering"]
        assert len(clustering) == 0


# ==================================================================
# Method 2: Temporal Aggregation
# ==================================================================

class TestTemporalAggregation:
    def test_detects_temporal_split(self, detector):
        # 4 invoices of ~$7500 within 14 days → aggregate $30K > $25K threshold
        rows = [
            {"invoice_number": f"INV-TS{i}", "vendor_id": "V001",
             "vendor_name": "SplitCo",
             "total_amount": 7200 + i * 200,
             "invoice_date": date(2025, 3, 1 + i * 3)}
            for i in range(4)
        ]
        df = _make_df(rows)
        findings = detector.detect(df)
        temporal = [f for f in findings
                    if f.evidence.get("method") == "temporal_aggregation"]
        assert len(temporal) >= 1
        assert temporal[0].evidence["cluster_count"] >= 3

    def test_no_flag_when_below_all_thresholds(self, detector):
        rows = [
            {"invoice_number": f"INV-{i}", "vendor_id": "V001",
             "total_amount": 500, "invoice_date": date(2025, 1, i + 1)}
            for i in range(4)
        ]
        df = _make_df(rows)
        findings = detector.detect(df)
        temporal = [f for f in findings
                    if f.evidence.get("method") == "temporal_aggregation"]
        assert len(temporal) == 0

    def test_different_vendors_not_combined(self, detector):
        rows = [
            {"invoice_number": f"INV-A{i}", "vendor_id": "V001",
             "total_amount": 4000, "invoice_date": date(2025, 1, i + 1)}
            for i in range(3)
        ] + [
            {"invoice_number": f"INV-B{i}", "vendor_id": "V002",
             "total_amount": 4000, "invoice_date": date(2025, 1, i + 1)}
            for i in range(3)
        ]
        df = _make_df(rows)
        findings = detector.detect(df)
        temporal = [f for f in findings
                    if f.evidence.get("method") == "temporal_aggregation"]
        # Each vendor individually sums to $12K — above $5K but not $25K
        # Vendors are NOT combined
        for f in temporal:
            assert f.evidence["cluster_count"] <= 3


# ==================================================================
# Method 3: PO / Project-Linked Splits
# ==================================================================

class TestPOProjectLinked:
    def test_po_linked_split_detected(self, detector):
        # 4 invoices sharing a PO, each below $5K but summing above it
        rows = [
            {"invoice_number": f"INV-PO{i}", "vendor_id": "V001",
             "vendor_name": "LinkedCo", "total_amount": 4500,
             "invoice_date": date(2025, 3, i + 1),
             "po_number": "PO-SHARED",
             "line_item_description": "Equipment supply batch"}
            for i in range(4)
        ]
        df = _make_df(rows)
        findings = detector.detect(df)
        linked = [f for f in findings
                  if f.evidence.get("method") == "po_project_linked"]
        assert len(linked) >= 1
        assert linked[0].evidence["link_key"] == "PO-SHARED"
        assert linked[0].severity == Severity.REVIEW

    def test_project_code_linked(self, detector):
        rows = [
            {"invoice_number": f"INV-P{i}", "vendor_id": "V001",
             "vendor_name": "ProjCo", "total_amount": 4800,
             "invoice_date": date(2025, 2, i + 1),
             "project_code": "PROJ-42"}
            for i in range(3)
        ]
        df = _make_df(rows)
        findings = detector.detect(df)
        linked = [f for f in findings
                  if f.evidence.get("method") == "po_project_linked"]
        assert len(linked) >= 1

    def test_different_pos_not_linked(self, detector):
        rows = [
            {"invoice_number": f"INV-{i}", "vendor_id": "V001",
             "total_amount": 4500,
             "invoice_date": date(2025, 1, i + 1),
             "po_number": f"PO-{i}"}
            for i in range(3)
        ]
        df = _make_df(rows)
        findings = detector.detect(df)
        linked = [f for f in findings
                  if f.evidence.get("method") == "po_project_linked"]
        assert len(linked) == 0

    def test_single_invoice_per_po_not_flagged(self, detector):
        rows = [
            {"invoice_number": "INV-001", "vendor_id": "V001",
             "total_amount": 4500, "po_number": "PO-X",
             "invoice_date": "2025-01-15"}
        ]
        df = _make_df(rows)
        findings = detector.detect(df)
        linked = [f for f in findings
                  if f.evidence.get("method") == "po_project_linked"]
        assert len(linked) == 0


# ==================================================================
# Method 4: Benford's Law
# ==================================================================

class TestBenfordsAnomaly:
    def test_detects_non_benford_distribution(self, detector):
        # All amounts start with 4 or 5 — highly non-Benford
        import numpy as np
        rng = np.random.default_rng(42)
        rows = []
        for i in range(60):
            rows.append({
                "invoice_number": f"INV-{i:03d}", "vendor_id": "V001",
                "vendor_name": "FakeDistro",
                "total_amount": float(rng.integers(4000, 5999)),
                "invoice_date": date(2024, (i % 12) + 1, 15),
            })
        df = _make_df(rows)
        findings = detector.detect(df)
        benford = [f for f in findings
                   if f.evidence.get("method") == "benfords_anomaly"]
        assert len(benford) >= 1
        assert benford[0].severity == Severity.INFORMATIONAL

    def test_too_few_invoices_skipped(self, detector):
        rows = [
            {"invoice_number": f"INV-{i}", "vendor_id": "V001",
             "total_amount": 4500 + i,
             "invoice_date": date(2024, (i % 12) + 1, 15)}
            for i in range(10)
        ]
        df = _make_df(rows)
        findings = detector.detect(df)
        benford = [f for f in findings
                   if f.evidence.get("method") == "benfords_anomaly"]
        assert len(benford) == 0


# ==================================================================
# Method 5: Pattern Change
# ==================================================================

class TestPatternChange:
    def test_average_drop_with_maintained_spend(self, detector):
        rows = []
        # Prior quarters: 3 invoices/quarter at $10K each
        for q in range(4):
            for i in range(3):
                month = q * 3 + (i % 3) + 1
                rows.append({
                    "invoice_number": f"INV-P{q}{i}", "vendor_id": "V001",
                    "vendor_name": "DriftCo",
                    "total_amount": 10000,
                    "invoice_date": date(2024, month, 15),
                })
        # Current quarter: 10 invoices at $3K each (avg dropped 70%, total $30K ≈ prior $30K)
        for i in range(10):
            rows.append({
                "invoice_number": f"INV-C{i}", "vendor_id": "V001",
                "vendor_name": "DriftCo",
                "total_amount": 3000,
                "invoice_date": date(2025, 1 + (i % 3), 15),
            })
        df = _make_df(rows)
        findings = detector.detect(df)
        pattern = [f for f in findings
                   if f.evidence.get("method") == "pattern_change"
                   and f.evidence.get("subtype") == "average_drop"]
        assert len(pattern) >= 1
        assert "dropped" in pattern[0].description.lower()

    def test_frequency_increase(self, detector):
        rows = []
        # Prior quarters: 2 invoices per quarter
        for q in range(4):
            for i in range(2):
                month = q * 3 + i + 1
                rows.append({
                    "invoice_number": f"INV-P{q}{i}", "vendor_id": "V001",
                    "vendor_name": "FreqCo",
                    "total_amount": 5000,
                    "invoice_date": date(2024, month, 15),
                })
        # Current quarter: 6 invoices (3x frequency) but similar total
        for i in range(6):
            rows.append({
                "invoice_number": f"INV-F{i}", "vendor_id": "V001",
                "vendor_name": "FreqCo",
                "total_amount": 1500,
                "invoice_date": date(2025, 1 + (i % 3), 15),
            })
        df = _make_df(rows)
        findings = detector.detect(df)
        freq = [f for f in findings
                if f.evidence.get("method") == "pattern_change"
                and f.evidence.get("subtype") == "frequency_increase"]
        assert len(freq) >= 1

    def test_no_finding_consistent_pattern(self, detector):
        rows = []
        for q in range(5):
            for i in range(3):
                month = q * 3 + i + 1
                if month > 12:
                    continue
                rows.append({
                    "invoice_number": f"INV-{q}{i}", "vendor_id": "V001",
                    "vendor_name": "SteadyCo",
                    "total_amount": 5000,
                    "invoice_date": date(2024, month, 15),
                })
        df = _make_df(rows)
        findings = detector.detect(df)
        pattern = [f for f in findings
                   if f.evidence.get("method") == "pattern_change"]
        assert len(pattern) == 0


# ==================================================================
# Edge cases
# ==================================================================

class TestEdgeCases:
    def test_empty_dataframe(self, detector):
        df = pd.DataFrame(columns=[
            "invoice_id", "vendor_id", "total_amount", "invoice_date",
        ])
        findings = detector.detect(df)
        assert findings == []

    def test_single_vendor_single_invoice(self, detector):
        df = _make_df([{
            "invoice_number": "INV-001", "vendor_id": "V001",
            "total_amount": 4999, "invoice_date": "2025-01-15",
        }])
        findings = detector.detect(df)
        assert len(findings) == 0

    def test_stats_populated(self, detector):
        df = _make_df([
            {"invoice_number": f"INV-{i}", "vendor_id": "V001",
             "total_amount": 4000, "invoice_date": f"2025-01-{i+1:02d}"}
            for i in range(5)
        ])
        detector.detect(df)
        stats = detector.get_stats()
        assert stats["invoices_analyzed"] == 5
        assert stats["module"] == "split_invoicing"


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
        thresh_path = base / "sample_invoices" / "approval_thresholds.json"

        if not inv_path.exists():
            pytest.skip("Sample data not generated")

        df = pd.read_csv(inv_path, dtype=str, keep_default_na=False)
        df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
        if "invoice_id" not in df.columns:
            df["invoice_id"] = df["invoice_number"]

        with open(key_path) as f:
            fraud_key = json.load(f)
        thresholds = []
        if thresh_path.exists():
            with open(thresh_path) as f:
                thresholds = json.load(f)

        return df, fraud_key, thresholds

    def test_detects_split_invoicing_from_sample(
        self, config, cache, mock_llm, sample_data,
    ):
        df, fraud_key, thresholds = sample_data
        det = SplitInvoiceDetector(config, cache, mock_llm)
        findings = det.detect(
            df, supplementary_data={"approval_thresholds": thresholds},
        )

        split_frauds = [
            f for f in fraud_key["fraud_items"]
            if f["fraud_type"] == "split_invoicing"
        ]
        assert len(split_frauds) == 4

        # Check that at least some findings overlap with injected fraud
        fraud_inv_ids: set[str] = set()
        for sf in split_frauds:
            fraud_inv_ids.update(sf["invoice_ids"])

        detected_inv_ids: set[str] = set()
        for f in findings:
            detected_inv_ids.update(f.invoice_ids)

        overlap = fraud_inv_ids & detected_inv_ids
        assert len(overlap) >= 3, (
            f"Expected to detect at least 3 fraud invoices, "
            f"found {len(overlap)} overlap out of {len(fraud_inv_ids)}"
        )

    def test_multiple_methods_fire(self, config, cache, mock_llm, sample_data):
        df, _, thresholds = sample_data
        det = SplitInvoiceDetector(config, cache, mock_llm)
        findings = det.detect(
            df, supplementary_data={"approval_thresholds": thresholds},
        )
        methods = {f.evidence.get("method") for f in findings}
        assert len(methods) >= 2, (
            f"Expected at least 2 detection methods to fire, got: {methods}"
        )
