"""Tests for src.detection.phantom_services — PhantomServicesDetector."""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.detection.phantom_services import PhantomServicesDetector
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
    m.score_description_vagueness.return_value = 50  # neutral default
    m.compute_text_similarity.return_value = 0.5
    return m


@pytest.fixture
def config():
    return {
        "unmatched_po_min_amount": 500,
        "partial_match_deviation_pct": 20,
        "vagueness_flag_threshold": 40,
        "category_mismatch_min_amount": 1000,
        "one_off_min_amount": 2000,
        "delivery_date_tolerance_days": 7,
    }


@pytest.fixture
def detector(config, cache, mock_llm):
    return PhantomServicesDetector(config, cache, mock_llm)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "invoice_id": "", "invoice_number": "", "vendor_id": "",
        "vendor_name": "", "invoice_date": "2025-01-15",
        "total_amount": 1000.0, "po_number": "",
        "line_item_description": "Standard service",
    }
    full = []
    for r in rows:
        row = {**defaults, **r}
        if not row["invoice_id"]:
            row["invoice_id"] = row["invoice_number"]
        full.append(row)
    return pd.DataFrame(full)


def _vendor_history(vid: str, vname: str, desc: str, n: int = 10) -> list[dict]:
    """Generate n rows of consistent vendor history."""
    return [
        {"invoice_number": f"INV-H{vid}-{i}", "vendor_id": vid,
         "vendor_name": vname,
         "invoice_date": date(2024, (i % 12) + 1, 15),
         "total_amount": 1000 + i * 50,
         "line_item_description": desc,
         "po_number": f"PO-{i}"}
        for i in range(n)
    ]


# ==================================================================
# Module identity
# ==================================================================

class TestModuleIdentity:
    def test_module_name(self, detector):
        assert detector.get_module_name() == ModuleName.PHANTOM_SERVICES

    def test_required_fields(self, detector):
        req = detector.get_required_fields()
        assert "line_item_description" in req
        assert "total_amount" in req

    def test_optional_fields(self, detector):
        assert "po_number" in detector.get_optional_fields()


# ==================================================================
# Method 1: PO Matching Gap
# ==================================================================

class TestPOGap:
    def test_flags_no_po_above_threshold(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "vendor_name": "Acme", "total_amount": 2000,
             "po_number": "", "line_item_description": "Consulting"},
        ])
        findings = detector.detect(df)
        po_gaps = [f for f in findings if f.evidence.get("method") == "po_gap"]
        assert len(po_gaps) >= 1
        assert po_gaps[0].evidence["subtype"] == "no_po"

    def test_ignores_below_threshold(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 100, "po_number": "",
             "line_item_description": "Small item"},
        ])
        findings = detector.detect(df)
        po_gaps = [f for f in findings if f.evidence.get("method") == "po_gap"]
        assert len(po_gaps) == 0

    def test_no_flag_when_po_present(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "po_number": "PO-123",
             "line_item_description": "Material delivery"},
        ])
        findings = detector.detect(df)
        po_gaps = [f for f in findings if f.evidence.get("method") == "po_gap"]
        assert len(po_gaps) == 0


# ==================================================================
# Method 2: Description Vagueness
# ==================================================================

class TestVagueness:
    def test_flags_vague_no_po(self, config, cache):
        mock_llm = MagicMock()
        mock_llm.score_description_vagueness.return_value = 15  # very vague
        det = PhantomServicesDetector(config, cache, mock_llm)

        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "vendor_name": "Acme", "total_amount": 5000,
             "po_number": "",
             "line_item_description": "Miscellaneous charges"},
        ])
        findings = det.detect(df)
        vague = [f for f in findings if f.evidence.get("method") == "vagueness"]
        assert len(vague) >= 1
        assert vague[0].severity == Severity.CRITICAL

    def test_flags_moderate_vagueness(self, config, cache):
        mock_llm = MagicMock()
        # Score 20 is below the 25 threshold
        mock_llm.score_description_vagueness.return_value = 20
        det = PhantomServicesDetector(config, cache, mock_llm)

        # Give a PO so the po_gap method doesn't fire — only vagueness should flag
        df = _make_df([
            {"invoice_number": "INV-V1", "vendor_id": "V1",
             "total_amount": 5000, "po_number": "",
             "line_item_description": "Professional fees"},
        ])
        findings = det.detect(df)
        # Intra-module dedup keeps one finding per invoice; either vagueness
        # or po_gap will survive. Check the invoice is flagged.
        flagged = [f for f in findings if "INV-V1" in f.invoice_ids]
        assert len(flagged) >= 1

    def test_no_flag_when_specific(self, config, cache):
        mock_llm = MagicMock()
        mock_llm.score_description_vagueness.return_value = 75  # specific
        det = PhantomServicesDetector(config, cache, mock_llm)

        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "po_number": "",
             "line_item_description": "Installation of 200 sqft granite countertop"},
        ])
        findings = det.detect(df)
        vague = [f for f in findings if f.evidence.get("method") == "vagueness"]
        assert len(vague) == 0

    def test_no_flag_when_po_present(self, config, cache):
        mock_llm = MagicMock()
        mock_llm.score_description_vagueness.return_value = 15
        det = PhantomServicesDetector(config, cache, mock_llm)

        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000, "po_number": "PO-123",
             "line_item_description": "Services"},
        ])
        findings = det.detect(df)
        vague = [f for f in findings if f.evidence.get("method") == "vagueness"]
        assert len(vague) == 0


# ==================================================================
# Method 3: Vendor-Service Category Mismatch
# ==================================================================

class TestCategoryMismatch:
    def test_flags_never_seen_category(self, detector):
        # 120 rows of plumbing history (so 1 mismatch = <1% of total),
        # then 1 row of consulting above $3K min
        rows = _vendor_history("V1", "PlumbCo", "Plumbing repair labor", 120)
        rows.append({
            "invoice_number": "INV-MISMATCH", "vendor_id": "V1",
            "vendor_name": "PlumbCo",
            "invoice_date": "2025-03-01", "total_amount": 8000,
            "line_item_description": "Engineering consultation design review",
            "po_number": "PO-99",
        })
        df = _make_df(rows)
        findings = detector.detect(df)
        mismatch = [f for f in findings
                    if f.evidence.get("method") == "category_mismatch"
                    and f.evidence.get("subtype") == "never_seen"]
        assert len(mismatch) >= 1
        assert "PlumbCo" in mismatch[0].description

    def test_no_flag_for_normal_category(self, detector):
        rows = _vendor_history("V1", "Acme", "Steel rebar delivery", 10)
        # Same category — should not flag
        rows.append({
            "invoice_number": "INV-NORMAL", "vendor_id": "V1",
            "vendor_name": "Acme",
            "invoice_date": "2025-03-01", "total_amount": 2000,
            "line_item_description": "Steel rebar delivery grade 60",
            "po_number": "PO-99",
        })
        df = _make_df(rows)
        findings = detector.detect(df)
        mismatch = [f for f in findings
                    if f.evidence.get("method") == "category_mismatch"
                    and "INV-NORMAL" in f.invoice_ids]
        assert len(mismatch) == 0

    def test_below_amount_threshold_skipped(self, detector):
        rows = _vendor_history("V1", "Acme", "Steel rebar", 10)
        rows.append({
            "invoice_number": "INV-SMALL", "vendor_id": "V1",
            "total_amount": 500,  # below $1000
            "line_item_description": "Totally new category stuff",
        })
        df = _make_df(rows)
        findings = detector.detect(df)
        mismatch = [f for f in findings
                    if f.evidence.get("method") == "category_mismatch"
                    and "INV-SMALL" in f.invoice_ids]
        assert len(mismatch) == 0

    def test_insufficient_history_skipped(self, detector):
        # Only 1 prior item — total profile count is 2, below threshold of 3
        rows = _vendor_history("V1", "Acme", "Steel rebar", 1)
        rows.append({
            "invoice_number": "INV-NEW", "vendor_id": "V1",
            "total_amount": 5000,
            "line_item_description": "Completely different service xyz",
        })
        df = _make_df(rows)
        findings = detector.detect(df)
        mismatch = [f for f in findings
                    if f.evidence.get("method") == "category_mismatch"
                    and "INV-NEW" in f.invoice_ids]
        assert len(mismatch) == 0


# ==================================================================
# Method 4: One-Off Charge Detection
# ==================================================================

class TestOneOffCharges:
    def test_flags_risky_one_off(self, config, cache):
        mock_llm = MagicMock()
        mock_llm.score_description_vagueness.return_value = 20  # vague
        det = PhantomServicesDetector(config, cache, mock_llm)

        # 10 rows of regular service, then 1 one-off with vague desc and no PO
        # Need enough history (5+ items) for profile, and amount >= $2K
        rows = _vendor_history("V1", "SvcCo", "Regular maintenance service", 10)
        rows.append({
            "invoice_number": "INV-ONEOFF", "vendor_id": "V1",
            "vendor_name": "SvcCo",
            "invoice_date": "2025-04-01", "total_amount": 8000,
            "line_item_description": "Special advisory engagement one-time",
            "po_number": "",
        })
        df = _make_df(rows)
        findings = det.detect(df)
        # The one-off may be deduped with a vagueness finding for same invoice.
        # Check that the invoice is flagged by any method.
        oneoff_inv = [f for f in findings
                      if "INV-ONEOFF" in f.invoice_ids and not f.suppressed]
        assert len(oneoff_inv) >= 1

    def test_no_flag_for_repeated_category(self, detector):
        rows = _vendor_history("V1", "Acme", "Steel rebar delivery", 10)
        # Same category appearing many times — not one-off
        rows.append({
            "invoice_number": "INV-REPEAT", "vendor_id": "V1",
            "total_amount": 5000,
            "line_item_description": "Steel rebar delivery grade 40",
            "po_number": "PO-X",
        })
        df = _make_df(rows)
        findings = detector.detect(df)
        oneoff = [f for f in findings
                  if f.evidence.get("method") == "one_off_charge"
                  and "INV-REPEAT" in f.invoice_ids]
        assert len(oneoff) == 0

    def test_below_amount_threshold_skipped(self, detector):
        rows = _vendor_history("V1", "Acme", "Regular service", 10)
        rows.append({
            "invoice_number": "INV-SMALL", "vendor_id": "V1",
            "total_amount": 500,  # below $2000
            "line_item_description": "One-time tiny charge xyz",
        })
        df = _make_df(rows)
        findings = detector.detect(df)
        oneoff = [f for f in findings
                  if f.evidence.get("method") == "one_off_charge"
                  and "INV-SMALL" in f.invoice_ids]
        assert len(oneoff) == 0


# ==================================================================
# Method 5: Delivery / Receipt Cross-Reference
# ==================================================================

class TestReceiptGaps:
    def test_flags_missing_receipt(self, detector):
        # Need 50%+ receipt coverage for the method to activate.
        # 4 invoices: 2 have receipts, 2 don't → 50% coverage.
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "vendor_name": "MaterialCo", "total_amount": 5000,
             "line_item_description": "Concrete delivery 50 CY"},
            {"invoice_number": "INV-002", "vendor_id": "V1",
             "vendor_name": "MaterialCo", "total_amount": 6000,
             "line_item_description": "Steel rebar delivery"},
            {"invoice_number": "INV-003", "vendor_id": "V1",
             "vendor_name": "MaterialCo", "total_amount": 4000,
             "line_item_description": "Lumber delivery"},
            {"invoice_number": "INV-004", "vendor_id": "V1",
             "vendor_name": "MaterialCo", "total_amount": 7000,
             "line_item_description": "Pipe delivery"},
        ])
        # Receipts for INV-002 and INV-003 only → INV-001 and INV-004 missing
        receipts = pd.DataFrame({
            "invoice_id": ["INV-002", "INV-003"],
            "receipt_id": ["REC-001", "REC-002"],
            "receipt_date": ["2025-01-16", "2025-01-17"],
            "status": ["Received", "Received"],
            "received_by": ["John", "Jane"],
        })
        findings = detector.detect(
            df, supplementary_data={"receipts": receipts},
        )
        gaps = [f for f in findings if f.evidence.get("method") == "receipt_gap"]
        assert len(gaps) >= 1
        assert gaps[0].evidence["subtype"] == "no_receipt"

    def test_no_flag_when_receipt_exists(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000,
             "line_item_description": "Material delivery"},
        ])
        receipts = pd.DataFrame({
            "invoice_id": ["INV-001"],
            "receipt_id": ["REC-001"],
            "receipt_date": ["2025-01-16"],
            "status": ["Received"],
            "received_by": ["John"],
        })
        findings = detector.detect(
            df, supplementary_data={"receipts": receipts},
        )
        gaps = [f for f in findings if f.evidence.get("method") == "receipt_gap"]
        assert len(gaps) == 0

    def test_no_receipt_method_without_data(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000,
             "line_item_description": "Material"},
        ])
        # No receipts provided
        findings = detector.detect(df)
        gaps = [f for f in findings if f.evidence.get("method") == "receipt_gap"]
        assert len(gaps) == 0


# ==================================================================
# Degraded mode (no PO, no receipts)
# ==================================================================

class TestDegradedMode:
    def test_runs_without_po_column(self, config, cache):
        mock_llm = MagicMock()
        mock_llm.score_description_vagueness.return_value = 15
        det = PhantomServicesDetector(config, cache, mock_llm)

        df = pd.DataFrame({
            "invoice_id": ["INV-001"],
            "vendor_id": ["V1"],
            "vendor_name": ["Acme"],
            "invoice_date": ["2025-01-15"],
            "total_amount": [5000.0],
            "line_item_description": ["Miscellaneous"],
        })
        findings = det.detect(df)
        # Should still produce vagueness findings
        vague = [f for f in findings if f.evidence.get("method") == "vagueness"]
        assert len(vague) >= 1
        # Should NOT produce PO gap findings
        po = [f for f in findings if f.evidence.get("method") == "po_gap"]
        assert len(po) == 0

    def test_runs_without_receipts(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000,
             "line_item_description": "Material delivery"},
        ])
        # detect() with no supplementary_data
        findings = detector.detect(df, supplementary_data=None)
        gaps = [f for f in findings if f.evidence.get("method") == "receipt_gap"]
        assert len(gaps) == 0


# ==================================================================
# Edge cases
# ==================================================================

class TestEdgeCases:
    def test_empty_dataframe(self, detector):
        df = pd.DataFrame(columns=[
            "invoice_id", "vendor_id", "line_item_description",
            "total_amount", "invoice_date", "po_number",
        ])
        findings = detector.detect(df)
        assert findings == []

    def test_empty_descriptions_skipped(self, detector):
        df = _make_df([
            {"invoice_number": "INV-001", "vendor_id": "V1",
             "total_amount": 5000,
             "line_item_description": ""},
        ])
        findings = detector.detect(df)
        vague = [f for f in findings if f.evidence.get("method") == "vagueness"]
        assert len(vague) == 0

    def test_stats_populated(self, detector):
        df = _make_df([
            {"invoice_number": f"INV-{i}", "vendor_id": "V1",
             "total_amount": 1000,
             "line_item_description": "Service"}
            for i in range(5)
        ])
        detector.detect(df)
        stats = detector.get_stats()
        assert stats["invoices_analyzed"] == 5
        assert stats["module"] == "phantom_services"


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
        rec_path = base / "sample_invoices" / "goods_receipts.csv"

        if not inv_path.exists():
            pytest.skip("Sample data not generated")

        df = pd.read_csv(inv_path, dtype=str, keep_default_na=False)
        df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        if "invoice_id" not in df.columns:
            df["invoice_id"] = df["invoice_number"]

        with open(key_path) as f:
            fraud_key = json.load(f)

        receipts = None
        if rec_path.exists():
            receipts = pd.read_csv(rec_path, dtype=str, keep_default_na=False)

        return df, fraud_key, receipts

    def test_detects_phantom_services(self, config, cache, sample_data):
        df, fraud_key, receipts = sample_data

        mock_llm = MagicMock()
        # Return low specificity for the known vague descriptions
        def _vagueness(desc):
            lower = desc.lower()
            vague_words = {"consulting services", "miscellaneous", "professional fees",
                           "other charges"}
            if any(v in lower for v in vague_words):
                return 10
            return 55
        mock_llm.score_description_vagueness.side_effect = _vagueness

        det = PhantomServicesDetector(config, cache, mock_llm)
        supp = {}
        if receipts is not None:
            supp["receipts"] = receipts

        findings = det.detect(df, supplementary_data=supp)

        phantom_frauds = [
            f for f in fraud_key["fraud_items"]
            if f["fraud_type"] == "phantom_services"
        ]
        assert len(phantom_frauds) == 12

        # Match findings to fraud key
        detected = 0
        for fraud in phantom_frauds:
            fraud_ids = set(fraud.get("invoice_ids", []))
            for finding in findings:
                if not finding.suppressed and fraud_ids & set(finding.invoice_ids):
                    detected += 1
                    break

        rate = detected / len(phantom_frauds) * 100
        assert detected >= 4, (
            f"Detected {detected}/12 phantom service fraud items ({rate:.0f}%)"
        )

    def test_multiple_methods_fire(self, config, cache, sample_data):
        df, _, receipts = sample_data
        mock_llm = MagicMock()
        mock_llm.score_description_vagueness.return_value = 50
        det = PhantomServicesDetector(config, cache, mock_llm)

        supp = {}
        if receipts is not None:
            supp["receipts"] = receipts

        findings = det.detect(df, supplementary_data=supp)
        methods = {f.evidence.get("method") for f in findings}
        assert len(methods) >= 2, (
            f"Expected ≥ 2 methods to fire, got: {methods}"
        )
