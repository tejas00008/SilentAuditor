"""Tests for src.detection.price_creep — PriceCreepDetector."""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.detection.price_creep import PriceCreepDetector
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
        "min_data_points": 4,
        "min_months": 3,
        "cumulative_threshold_6mo_pct": 8,
        "cumulative_threshold_12mo_pct": 12,
        "single_jump_threshold_pct": 5,
        "contract_tolerance_pct": 3,
        "p_value_significance": 0.05,
    }


@pytest.fixture
def detector(config, cache, mock_llm):
    return PriceCreepDetector(config, cache, mock_llm)


def _make_series_df(
    vendor_id: str = "V001",
    vendor_name: str = "Acme",
    item_desc: str = "Steel rebar 40mm",
    prices: list[float] = None,
    start_year: int = 2024,
    start_month: int = 1,
) -> pd.DataFrame:
    """Build a DataFrame with a single vendor-item price series."""
    if prices is None:
        prices = [100.0] * 6
    rows = []
    for i, price in enumerate(prices):
        month = start_month + i
        year = start_year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        rows.append({
            "invoice_id": f"INV-{i:04d}",
            "invoice_number": f"INV-{i:04d}",
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "invoice_date": date(year, month, 15),
            "line_item_description": item_desc,
            "unit_price": price,
            "quantity": 10.0,
        })
    return pd.DataFrame(rows)


# ==================================================================
# Module identity
# ==================================================================

class TestModuleIdentity:
    def test_module_name(self, detector):
        assert detector.get_module_name() == ModuleName.PRICE_CREEP

    def test_required_fields(self, detector):
        assert "unit_price" in detector.get_required_fields()
        assert "line_item_description" in detector.get_required_fields()

    def test_optional_fields(self, detector):
        assert "quantity" in detector.get_optional_fields()


# ==================================================================
# Build time series
# ==================================================================

class TestBuildTimeSeries:
    def test_groups_by_vendor_item(self, detector):
        df = _make_series_df(prices=[100, 102, 104, 106, 108, 110])
        ts = detector._build_price_time_series(df)
        assert len(ts) == 1
        key = list(ts.keys())[0]
        assert key[0] == "V001"
        assert len(ts[key]) == 6

    def test_filters_short_series(self, detector):
        # Only 2 data points — below min_data_points (4)
        df = _make_series_df(prices=[100, 102])
        ts = detector._build_price_time_series(df)
        assert len(ts) == 0

    def test_filters_short_timeframe(self, detector):
        # 4 points but all in same month — below min_months (3)
        rows = []
        for i in range(4):
            rows.append({
                "invoice_id": f"INV-{i}", "invoice_number": f"INV-{i}",
                "vendor_id": "V001", "vendor_name": "Acme",
                "invoice_date": date(2024, 1, i + 1),
                "line_item_description": "Widget",
                "unit_price": 100 + i, "quantity": 10,
            })
        df = pd.DataFrame(rows)
        ts = detector._build_price_time_series(df)
        assert len(ts) == 0

    def test_multiple_vendors(self, detector):
        df1 = _make_series_df("V001", "Acme", "Widget", [100, 102, 104, 106, 108, 110])
        df2 = _make_series_df("V002", "Beta", "Gizmo", [50, 51, 52, 53, 54, 55])
        # Fix invoice IDs to not collide
        df2["invoice_id"] = [f"INV-B{i}" for i in range(len(df2))]
        df = pd.concat([df1, df2], ignore_index=True)
        ts = detector._build_price_time_series(df)
        assert len(ts) == 2

    def test_zero_prices_excluded(self, detector):
        df = _make_series_df(prices=[0, 0, 0, 0, 0, 0])
        ts = detector._build_price_time_series(df)
        assert len(ts) == 0


# ==================================================================
# Trend analysis
# ==================================================================

class TestAnalyzeTrend:
    def test_linear_increase(self, detector):
        series = [
            {"date": date(2024, m, 15), "unit_price": 100 + m * 5,
             "quantity": 10.0, "invoice_id": f"I{m}"}
            for m in range(1, 7)
        ]
        result = detector._analyze_trend(series)
        assert result["cumulative_change_pct"] > 0
        assert result["slope"] > 0
        assert result["data_points"] == 6

    def test_flat_prices(self, detector):
        series = [
            {"date": date(2024, m, 15), "unit_price": 100.0,
             "quantity": 10.0, "invoice_id": f"I{m}"}
            for m in range(1, 7)
        ]
        result = detector._analyze_trend(series)
        assert result["cumulative_change_pct"] == 0.0
        assert result["max_single_jump_pct"] == 0.0

    def test_single_jump(self, detector):
        prices = [100, 100, 100, 100, 112, 112]
        series = [
            {"date": date(2024, m, 15), "unit_price": prices[m - 1],
             "quantity": 10.0, "invoice_id": f"I{m}"}
            for m in range(1, 7)
        ]
        result = detector._analyze_trend(series)
        assert result["max_single_jump_pct"] == 12.0
        assert result["max_jump_date"] == date(2024, 5, 15)

    def test_acceleration(self, detector):
        # Each jump is bigger than the last: 1%, 2%, 4%
        prices = [100, 100, 101, 103.02, 107.14, 107.14]
        series = [
            {"date": date(2024, m, 15), "unit_price": prices[m - 1],
             "quantity": 10.0, "invoice_id": f"I{m}"}
            for m in range(1, 7)
        ]
        result = detector._analyze_trend(series)
        # Last 3 jumps: 0→1%, 1→~2%, ~2→~4%, then 0% — need to construct
        # specifically.  Let's use a cleaner series:
        prices2 = [100, 101, 103, 107, 115, 131]  # jumps: 1, ~2, ~3.9, ~7.5, ~13.9
        series2 = [
            {"date": date(2024, m, 15), "unit_price": prices2[m - 1],
             "quantity": 10.0, "invoice_id": f"I{m}"}
            for m in range(1, 7)
        ]
        result2 = detector._analyze_trend(series2)
        assert result2["is_accelerating"] is True

    def test_not_accelerating(self, detector):
        # Decreasing rate of change: 10%, 8%, 5% — not accelerating
        prices = [100, 110, 118.8, 124.74, 130.0, 134.0]
        series = [
            {"date": date(2024, m, 15), "unit_price": prices[m - 1],
             "quantity": 10.0, "invoice_id": f"I{m}"}
            for m in range(1, 7)
        ]
        result = detector._analyze_trend(series)
        assert result["is_accelerating"] is False


# ==================================================================
# Contract rate comparison
# ==================================================================

class TestContractRates:
    def test_breach_detected(self, detector):
        contracts = [{
            "vendor_id": "V001",
            "contract_start_date": "2024-01-01",
            "annual_escalation_pct": "3",
            "rates": [{"item_description": "Steel rebar", "rate": "100"}],
        }]
        # Current price 120 — base 100 + 3% escalation ≈ 103 allowed
        result = detector._check_contract_rates(
            "V001", "steel rebar 40mm", Decimal("120"), contracts,
        )
        assert result is not None
        assert result["deviation_pct"] > 10

    def test_no_breach_within_tolerance(self, detector):
        contracts = [{
            "vendor_id": "V001",
            "contract_start_date": "2024-01-01",
            "annual_escalation_pct": "3",
            "rates": [{"item_description": "Steel rebar", "rate": "100"}],
        }]
        # 103 is within allowed range
        result = detector._check_contract_rates(
            "V001", "steel rebar 40mm", Decimal("103"), contracts,
        )
        assert result is None

    def test_no_matching_contract(self, detector):
        contracts = [{
            "vendor_id": "V999",
            "rates": [{"item_description": "Something else", "rate": "100"}],
        }]
        result = detector._check_contract_rates(
            "V001", "steel rebar", Decimal("200"), contracts,
        )
        assert result is None

    def test_no_matching_item(self, detector):
        contracts = [{
            "vendor_id": "V001",
            "contract_start_date": "2024-01-01",
            "annual_escalation_pct": "0",
            "rates": [{"item_description": "Completely unrelated thing", "rate": "100"}],
        }]
        result = detector._check_contract_rates(
            "V001", "steel rebar", Decimal("200"), contracts,
        )
        assert result is None


# ==================================================================
# Volume changes
# ==================================================================

class TestVolumeChanges:
    def test_significant_change(self, detector):
        series = (
            [{"quantity": 10.0}] * 4
            + [{"quantity": 20.0}] * 4
        )
        note = detector._check_volume_changes(series)
        assert note is not None
        assert "increased" in note

    def test_no_significant_change(self, detector):
        series = [{"quantity": 10.0}] * 6
        note = detector._check_volume_changes(series)
        assert note is None

    def test_too_short(self, detector):
        series = [{"quantity": 10.0}] * 3
        assert detector._check_volume_changes(series) is None


# ==================================================================
# Full detection — cumulative drift
# ==================================================================

class TestDetectCumulativeDrift:
    def test_detects_gradual_increase(self, detector):
        # 10% increase over 6 months — exceeds 8% threshold
        prices = [100, 102, 104, 106, 108, 110]
        df = _make_series_df(prices=prices)
        findings = detector.detect(df)
        drift_findings = [f for f in findings
                          if f.evidence.get("pattern") == "cumulative_drift"]
        assert len(drift_findings) >= 1
        assert drift_findings[0].evidence["cumulative_change_pct"] == 10.0

    def test_no_finding_for_small_increase(self, detector):
        # 3% over 6 months — below 8% threshold
        prices = [100, 100.5, 101, 101.5, 102, 103]
        df = _make_series_df(prices=prices)
        findings = detector.detect(df)
        drift_findings = [f for f in findings
                          if f.evidence.get("pattern") == "cumulative_drift"]
        assert len(drift_findings) == 0

    def test_severity_critical_above_15pct(self, detector):
        prices = [100, 105, 110, 115, 120, 125]  # 25% cumulative
        df = _make_series_df(prices=prices)
        findings = detector.detect(df)
        drift = [f for f in findings
                 if f.evidence.get("pattern") == "cumulative_drift"]
        assert len(drift) >= 1
        assert drift[0].severity == Severity.CRITICAL

    def test_severity_review_8_to_15(self, detector):
        prices = [100, 102, 104, 106, 108, 110]  # 10%
        df = _make_series_df(prices=prices)
        findings = detector.detect(df)
        drift = [f for f in findings
                 if f.evidence.get("pattern") == "cumulative_drift"]
        if drift:
            assert drift[0].severity == Severity.REVIEW

    def test_amount_at_risk_computed(self, detector):
        prices = [100, 105, 110, 115, 120, 125]
        df = _make_series_df(prices=prices)
        findings = detector.detect(df)
        drift = [f for f in findings
                 if f.evidence.get("pattern") == "cumulative_drift"]
        assert len(drift) >= 1
        assert drift[0].amount_at_risk > Decimal("0")


# ==================================================================
# Full detection — single jump
# ==================================================================

class TestDetectSingleJump:
    def test_detects_large_jump(self, detector):
        # Flat then 6% spike — above 5% jump threshold but cumulative is
        # only 6%, below the 8% cumulative threshold for 6 months.
        # So only a single_jump finding should be emitted.
        prices = [100, 100, 100, 100, 106, 106]
        df = _make_series_df(prices=prices)
        findings = detector.detect(df)
        jumps = [f for f in findings
                 if f.evidence.get("pattern") == "single_jump"]
        assert len(jumps) >= 1
        assert jumps[0].evidence["jump_pct"] == 6.0

    def test_no_finding_for_small_jump(self, detector):
        prices = [100, 100, 100, 100, 103, 103]  # 3% < 5%
        df = _make_series_df(prices=prices)
        findings = detector.detect(df)
        jumps = [f for f in findings
                 if f.evidence.get("pattern") == "single_jump"]
        assert len(jumps) == 0


# ==================================================================
# Full detection — contract breach
# ==================================================================

class TestDetectContractBreach:
    def test_contract_breach_finding(self, detector):
        # Prices are flat at 120, but contract rate is 100
        prices = [120, 120, 120, 120, 120, 120]
        df = _make_series_df(prices=prices)
        contracts = [{
            "vendor_id": "V001",
            "contract_start_date": "2024-01-01",
            "annual_escalation_pct": "3",
            "rates": [{"item_description": "Steel rebar", "rate": "100"}],
        }]
        findings = detector.detect(df, supplementary_data={"contracts": contracts})
        breach = [f for f in findings
                  if f.evidence.get("pattern") == "contract_breach"]
        assert len(breach) >= 1
        assert breach[0].severity == Severity.CRITICAL


# ==================================================================
# Insufficient data
# ==================================================================

class TestInsufficientData:
    def test_too_few_points(self, detector):
        prices = [100, 200]  # only 2 points
        df = _make_series_df(prices=prices)
        findings = detector.detect(df)
        assert len(findings) == 0

    def test_too_short_timeframe(self, detector):
        rows = []
        for i in range(5):
            rows.append({
                "invoice_id": f"INV-{i}", "invoice_number": f"INV-{i}",
                "vendor_id": "V001", "vendor_name": "Acme",
                "invoice_date": date(2024, 1, i + 1),  # all in January
                "line_item_description": "Widget",
                "unit_price": 100 + i * 10, "quantity": 10,
            })
        df = pd.DataFrame(rows)
        findings = detector.detect(df)
        assert len(findings) == 0

    def test_empty_dataframe(self, detector):
        df = pd.DataFrame(columns=[
            "vendor_id", "line_item_description", "unit_price",
            "invoice_date", "invoice_id", "vendor_name", "quantity",
        ])
        findings = detector.detect(df)
        assert findings == []


# ==================================================================
# Stats
# ==================================================================

class TestStats:
    def test_stats_populated(self, detector):
        prices = [100, 102, 104, 106, 108, 110]
        df = _make_series_df(prices=prices)
        detector.detect(df)
        stats = detector.get_stats()
        assert stats["invoices_analyzed"] == 6
        assert stats["module"] == "price_creep"


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
        contracts_path = base / "sample_contracts" / "contracts.json"

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
        contracts = []
        if contracts_path.exists():
            with open(contracts_path) as f:
                contracts = json.load(f)
        return df, fraud_key, contracts

    def test_detects_price_creep_from_sample(self, config, cache, mock_llm, sample_data):
        df, fraud_key, contracts = sample_data
        det = PriceCreepDetector(config, cache, mock_llm)
        findings = det.detect(df, supplementary_data={"contracts": contracts})

        # Price creep items in fraud key
        creep_frauds = [
            f for f in fraud_key["fraud_items"]
            if f["fraud_type"] == "price_creep"
        ]
        assert len(creep_frauds) == 8

        # We should detect at least some of them
        creep_vendors = {f["vendor"] for f in creep_frauds}
        detected_vendors = {f.vendor_name for f in findings}
        overlap = creep_vendors & detected_vendors

        assert len(overlap) >= 1, (
            f"Expected to detect at least 1 of {len(creep_vendors)} "
            f"price-creep vendors, detected: {detected_vendors}"
        )
        assert len(findings) > 0
