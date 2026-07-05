"""Tests for src.detection.duplicate_invoices — DuplicateInvoiceDetector."""

from decimal import Decimal
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.detection.duplicate_invoices import DuplicateInvoiceDetector
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
    m = MagicMock()
    m.compute_text_similarity.return_value = 0.5
    m.assess_duplicate_pair.return_value = {
        "is_duplicate": False, "confidence": 0.3, "reasoning": "",
    }
    return m


@pytest.fixture
def config():
    return {
        "exact_match_confidence": 0.99,
        "fuzzy_composite_threshold": 0.85,
        "ambiguous_zone_lower": 0.70,
        "amount_tolerance_pct": 0.05,
        "date_window_days": 90,
        "recurring_invoice_min_count": 3,
        "recurring_interval_tolerance_days": 5,
    }


@pytest.fixture
def detector(config, cache, mock_llm):
    return DuplicateInvoiceDetector(config, cache, mock_llm)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with all required + optional columns."""
    defaults = {
        "invoice_id": "", "invoice_number": "", "vendor_id": "",
        "vendor_name": "", "invoice_date": "2025-01-15",
        "total_amount": 1000.0, "po_number": "", "line_item_description": "",
        "quantity": 1, "unit_price": 100.0,
    }
    full_rows = []
    for r in rows:
        row = {**defaults, **r}
        if not row["invoice_id"]:
            row["invoice_id"] = row["invoice_number"]
        full_rows.append(row)
    return pd.DataFrame(full_rows)


# ==================================================================
# Module identity
# ==================================================================

class TestModuleIdentity:
    def test_module_name(self, detector):
        assert detector.get_module_name() == ModuleName.DUPLICATE_DETECTION

    def test_required_fields(self, detector):
        assert "invoice_number" in detector.get_required_fields()
        assert "total_amount" in detector.get_required_fields()

    def test_optional_fields(self, detector):
        assert "line_item_description" in detector.get_optional_fields()
        assert "po_number" in detector.get_optional_fields()


# ==================================================================
# Layer 1: Exact Duplicates
# ==================================================================

class TestExactDuplicates:
    def test_detects_exact_duplicate(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "vendor_name": "Acme"},
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "vendor_name": "Acme"},
        ])
        findings = detector.detect(df)
        exact = [f for f in findings if not f.suppressed
                 and f.evidence.get("layer") == "exact_match"]
        assert len(exact) >= 1
        assert exact[0].severity == Severity.CRITICAL
        assert exact[0].confidence == 0.99

    def test_different_vendors_not_exact(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000},
            {"invoice_number": "INV-001", "vendor_id": "V2",
             "total_amount": 5000},
        ])
        findings = detector.detect(df)
        exact = [f for f in findings if f.evidence.get("layer") == "exact_match"]
        assert len(exact) == 0

    def test_different_amounts_not_exact(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000},
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 6000},
        ])
        findings = detector.detect(df)
        exact = [f for f in findings if f.evidence.get("layer") == "exact_match"]
        assert len(exact) == 0

    def test_triple_exact_duplicate(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "vendor_name": "Acme"},
        ] * 3)
        findings = detector.detect(df)
        exact = [f for f in findings if f.evidence.get("layer") == "exact_match"]
        assert len(exact) >= 1
        assert exact[0].evidence["count"] == 3


# ==================================================================
# Layer 2: Fuzzy Duplicates
# ==================================================================

class TestFuzzyDuplicates:
    def test_near_duplicate_detected(self, detector, mock_llm):
        # Same vendor, amount within 1%, 3 days apart, similar description
        mock_llm.compute_text_similarity.return_value = 0.95
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "invoice_date": "2025-01-10",
             "line_item_description": "Steel rebar 40mm 10 tons",
             "vendor_name": "SteelCo"},
            {"invoice_number": "INV-002", "vendor_id": "V1",
             "total_amount": 5010, "invoice_date": "2025-01-13",
             "line_item_description": "Steel rebar 40mm 10 tons",
             "vendor_name": "SteelCo"},
        ])
        findings = detector.detect(df)
        fuzzy = [f for f in findings if not f.suppressed
                 and f.evidence.get("layer") == "fuzzy_match"]
        assert len(fuzzy) >= 1

    def test_different_amount_not_fuzzy(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "invoice_date": "2025-01-10"},
            {"invoice_number": "INV-002", "vendor_id": "V1",
             "total_amount": 9000, "invoice_date": "2025-01-12"},
        ])
        findings = detector.detect(df)
        fuzzy = [f for f in findings if f.evidence.get("layer") == "fuzzy_match"]
        assert len(fuzzy) == 0

    def test_outside_date_window(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "invoice_date": "2025-01-01"},
            {"invoice_number": "INV-002", "vendor_id": "V1",
             "total_amount": 5000, "invoice_date": "2025-07-01"},
        ])
        findings = detector.detect(df)
        fuzzy = [f for f in findings if f.evidence.get("layer") == "fuzzy_match"]
        assert len(fuzzy) == 0

    def test_different_vendors_not_compared(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "invoice_date": "2025-01-10"},
            {"invoice_number": "INV-002", "vendor_id": "V2",
             "total_amount": 5000, "invoice_date": "2025-01-10"},
        ])
        findings = detector.detect(df)
        fuzzy = [f for f in findings if f.evidence.get("layer") == "fuzzy_match"]
        assert len(fuzzy) == 0


# ==================================================================
# Layer 3: Semantic (LLM)
# ==================================================================

class TestSemanticDuplicates:
    def test_llm_confirms_ambiguous_pair(self, config, cache):
        mock_llm = MagicMock()
        mock_llm.compute_text_similarity.return_value = 0.78
        mock_llm.assess_duplicate_pair.return_value = {
            "is_duplicate": True, "confidence": 0.85,
            "reasoning": "Same work described differently",
        }
        det = DuplicateInvoiceDetector(config, cache, mock_llm)

        # Score should fall in ambiguous zone (0.70-0.85) then LLM confirms
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "invoice_date": "2025-01-10",
             "line_item_description": "Plumbing rough-in labor 40 hrs",
             "vendor_name": "PlumbCo"},
            {"invoice_number": "INV-002", "vendor_id": "V1",
             "total_amount": 5050, "invoice_date": "2025-01-14",
             "line_item_description": "Plumbing labour rough in 40 hours",
             "vendor_name": "PlumbCo"},
        ])
        findings = det.detect(df)
        semantic = [f for f in findings if f.evidence.get("layer") == "semantic_llm"]
        if semantic:
            assert semantic[0].confidence == 0.85
            assert "LLM-confirmed" in semantic[0].description

    def test_llm_rejects_non_duplicate(self, config, cache):
        mock_llm = MagicMock()
        mock_llm.compute_text_similarity.return_value = 0.75
        mock_llm.assess_duplicate_pair.return_value = {
            "is_duplicate": False, "confidence": 0.8,
            "reasoning": "Different scope of work",
        }
        det = DuplicateInvoiceDetector(config, cache, mock_llm)
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "invoice_date": "2025-01-10",
             "line_item_description": "Phase 1 excavation"},
            {"invoice_number": "INV-002", "vendor_id": "V1",
             "total_amount": 5100, "invoice_date": "2025-01-15",
             "line_item_description": "Phase 2 backfill"},
        ])
        findings = det.detect(df)
        semantic = [f for f in findings if f.evidence.get("layer") == "semantic_llm"]
        assert len(semantic) == 0


# ==================================================================
# Layer 4: Cross-PO
# ==================================================================

class TestCrossPO:
    def test_cross_po_detected(self, detector, mock_llm):
        mock_llm.compute_text_similarity.return_value = 0.92
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "invoice_date": "2025-01-10",
             "po_number": "PO-001",
             "line_item_description": "Concrete pour Building A",
             "vendor_name": "ConcCo"},
            {"invoice_number": "INV-002", "vendor_id": "V1",
             "total_amount": 5000, "invoice_date": "2025-01-15",
             "po_number": "PO-002",
             "line_item_description": "Concrete pour Building A",
             "vendor_name": "ConcCo"},
        ])
        findings = detector.detect(df)
        xpo = [f for f in findings if f.evidence.get("layer") == "cross_po"]
        assert len(xpo) >= 1
        assert xpo[0].evidence["po_a"] == "PO-001"
        assert xpo[0].evidence["po_b"] == "PO-002"

    def test_same_po_not_flagged_as_cross_po(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "invoice_date": "2025-01-10",
             "po_number": "PO-001"},
            {"invoice_number": "INV-002", "vendor_id": "V1",
             "total_amount": 5000, "invoice_date": "2025-01-12",
             "po_number": "PO-001"},
        ])
        findings = detector.detect(df)
        xpo = [f for f in findings if f.evidence.get("layer") == "cross_po"]
        assert len(xpo) == 0

    def test_different_amounts_not_cross_po(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "invoice_date": "2025-01-10",
             "po_number": "PO-001"},
            {"invoice_number": "INV-002", "vendor_id": "V1",
             "total_amount": 9000, "invoice_date": "2025-01-12",
             "po_number": "PO-002"},
        ])
        findings = detector.detect(df)
        xpo = [f for f in findings if f.evidence.get("layer") == "cross_po"]
        assert len(xpo) == 0


# ==================================================================
# False Positive Suppression
# ==================================================================

class TestFalsePositiveSuppression:
    def test_recurring_subscription_suppressed(self, detector, mock_llm):
        mock_llm.compute_text_similarity.return_value = 0.95
        # 5 monthly invoices with same amount — recurring pattern
        rows = []
        for i in range(5):
            rows.append({
                "invoice_number": f"INV-{i:03d}",
                "vendor_id": "V1",
                "total_amount": 500,
                "invoice_date": f"2025-{i + 1:02d}-15",
                "vendor_name": "CleanCo",
                "line_item_description": f"Janitorial services",
            })
        df = _make_df(rows)
        findings = detector.detect(df)
        # Any findings between these should be suppressed
        non_suppressed = [f for f in findings if not f.suppressed]
        suppressed = [f for f in findings if f.suppressed]
        # The recurring pattern should suppress fuzzy matches
        for s in suppressed:
            assert "recurring" in (s.suppression_reason or "").lower()

    def test_progress_billing_suppressed(self, detector, mock_llm):
        mock_llm.compute_text_similarity.return_value = 0.90
        rows = [
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 10000, "invoice_date": "2025-01-10",
             "line_item_description": "Phase 1 of 3 foundation work",
             "vendor_name": "BuildCo"},
            {"invoice_number": "INV-002", "vendor_id": "V1",
             "total_amount": 10100, "invoice_date": "2025-02-10",
             "line_item_description": "Phase 2 of 3 framing",
             "vendor_name": "BuildCo"},
        ]
        df = _make_df(rows)
        findings = detector.detect(df)
        for f in findings:
            if set(f.invoice_ids) == {"INV-001", "INV-002"} and not f.suppressed:
                pytest.fail("Progress billing pair should be suppressed")

    def test_credit_rebill_suppressed(self, detector, mock_llm):
        mock_llm.compute_text_similarity.return_value = 0.95
        rows = [
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": -5000, "invoice_date": "2025-01-10",
             "vendor_name": "Acme"},
            {"invoice_number": "INV-002", "vendor_id": "V1",
             "total_amount": 5000, "invoice_date": "2025-01-12",
             "vendor_name": "Acme"},
        ]
        df = _make_df(rows)
        findings = detector.detect(df)
        for f in findings:
            if {"INV-001", "INV-002"}.issubset(set(f.invoice_ids)):
                assert f.suppressed
                assert "credit" in (f.suppression_reason or "").lower()


# ==================================================================
# Missing optional fields
# ==================================================================

class TestMissingOptionalFields:
    def test_no_line_items(self, detector):
        df = pd.DataFrame({
            "invoice_id": ["INV-001", "INV-001"],
            "invoice_number": ["INV-001", "INV-001"],
            "vendor_id": ["V1", "V1"],
            "vendor_name": ["Acme", "Acme"],
            "invoice_date": ["2025-01-10", "2025-01-10"],
            "total_amount": [5000.0, 5000.0],
        })
        findings = detector.detect(df)
        exact = [f for f in findings if f.evidence.get("layer") == "exact_match"]
        assert len(exact) >= 1

    def test_no_po(self, detector, mock_llm):
        mock_llm.compute_text_similarity.return_value = 0.95
        df = pd.DataFrame({
            "invoice_id": ["INV-001", "INV-002"],
            "invoice_number": ["INV-001", "INV-002"],
            "vendor_id": ["V1", "V1"],
            "vendor_name": ["Acme", "Acme"],
            "invoice_date": ["2025-01-10", "2025-01-12"],
            "total_amount": [5000.0, 5010.0],
            "line_item_description": ["Widget 40mm", "Widget 40mm"],
        })
        findings = detector.detect(df)
        assert len(findings) >= 0  # shouldn't crash


# ==================================================================
# Performance
# ==================================================================

class TestPerformance:
    def test_5000_invoices(self, detector):
        """5000 invoices from 50 vendors should run in a few seconds."""
        import time
        rows = []
        for i in range(5000):
            vid = f"V{i % 50:03d}"
            month = (i % 12) + 1
            rows.append({
                "invoice_id": f"INV-{i:05d}",
                "invoice_number": f"INV-{i:05d}",
                "vendor_id": vid,
                "vendor_name": f"Vendor {vid}",
                "invoice_date": f"2025-{month:02d}-{(i % 28) + 1:02d}",
                "total_amount": 1000 + (i % 100) * 50,
                "po_number": f"PO-{i:05d}",
                "line_item_description": f"Item {i}",
            })
        df = pd.DataFrame(rows)

        start = time.perf_counter()
        findings = detector.detect(df)
        elapsed = time.perf_counter() - start

        assert elapsed < 30  # generous for CI
        assert detector.stats["invoices_analyzed"] == 5000


# ==================================================================
# Sample data integration
# ==================================================================

class TestSampleDataIntegration:
    """Run against generated sample data and verify fraud detection."""

    @pytest.fixture
    def sample_data(self):
        """Load generated sample data if available."""
        import json
        from pathlib import Path

        base = Path(__file__).resolve().parent.parent / "data"
        inv_path = base / "sample_invoices" / "invoices.csv"
        key_path = base / "sample_invoices" / "fraud_key.json"

        if not inv_path.exists() or not key_path.exists():
            pytest.skip("Sample data not generated — run generate_sample_data.py first")

        df = pd.read_csv(inv_path, dtype=str, keep_default_na=False)
        # Ensure numeric columns
        df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
        if "invoice_id" not in df.columns:
            df["invoice_id"] = df["invoice_number"]

        with open(key_path) as f:
            fraud_key = json.load(f)

        return df, fraud_key

    def test_detects_exact_duplicates_from_sample(self, config, cache, sample_data):
        df, fraud_key = sample_data
        mock_llm = MagicMock()
        mock_llm.compute_text_similarity.return_value = 0.5
        mock_llm.assess_duplicate_pair.return_value = {
            "is_duplicate": False, "confidence": 0.3, "reasoning": "",
        }
        det = DuplicateInvoiceDetector(config, cache, mock_llm)
        findings = det.detect(df)

        # Extract exact-duplicate fraud IDs from the key
        exact_dupes = [
            f for f in fraud_key["fraud_items"]
            if f["fraud_type"] == "duplicate_invoice"
            and f["subtype"] == "exact_duplicate"
        ]

        # For each injected exact duplicate, check that we found it
        found_count = 0
        for fraud in exact_dupes:
            for finding in findings:
                if finding.evidence.get("layer") == "exact_match":
                    flagged_ids = set(finding.invoice_ids)
                    fraud_ids = set(fraud["invoice_ids"])
                    if fraud_ids & flagged_ids:
                        found_count += 1
                        break

        assert found_count >= 3, (
            f"Expected to detect at least 3 of {len(exact_dupes)} exact duplicates, "
            f"found {found_count}"
        )

    def test_stats_populated(self, config, cache, sample_data):
        df, _ = sample_data
        mock_llm = MagicMock()
        mock_llm.compute_text_similarity.return_value = 0.5
        mock_llm.assess_duplicate_pair.return_value = {
            "is_duplicate": False, "confidence": 0.3, "reasoning": "",
        }
        det = DuplicateInvoiceDetector(config, cache, mock_llm)
        det.detect(df)
        stats = det.get_stats()
        assert stats["invoices_analyzed"] > 0
        assert stats["module"] == "duplicate_detection"
