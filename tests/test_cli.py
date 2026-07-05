"""Tests for src.main — CLI commands via Click CliRunner."""

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.main import cli

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_INVOICES_CSV = _DATA_DIR / "sample_invoices" / "invoices.csv"
_VENDORS_CSV = _DATA_DIR / "sample_vendor_master" / "vendors.csv"
_CONTRACTS_JSON = _DATA_DIR / "sample_contracts" / "contracts.json"
_THRESHOLDS_JSON = _DATA_DIR / "sample_invoices" / "approval_thresholds.json"
_RECEIPTS_CSV = _DATA_DIR / "sample_invoices" / "goods_receipts.csv"
_FRAUD_KEY = _DATA_DIR / "sample_invoices" / "fraud_key.json"


@pytest.fixture
def runner():
    return CliRunner()


def _has_sample_data() -> bool:
    return _INVOICES_CSV.exists()


# ==================================================================
# Help / basic invocation
# ==================================================================

class TestBasic:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "SilentAuditor" in result.output

    def test_analyze_help(self, runner):
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output

    def test_quality_check_help(self, runner):
        result = runner.invoke(cli, ["quality-check", "--help"])
        assert result.exit_code == 0

    def test_validate_help(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0

    def test_feedback_help(self, runner):
        result = runner.invoke(cli, ["feedback", "--help"])
        assert result.exit_code == 0

    def test_generate_sample_data_help(self, runner):
        result = runner.invoke(cli, ["generate-sample-data", "--help"])
        assert result.exit_code == 0


# ==================================================================
# Error handling
# ==================================================================

class TestErrorHandling:
    def test_missing_input_file(self, runner):
        result = runner.invoke(cli, ["analyze", "-i", "/nonexistent/file.csv"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_quality_check_missing_file(self, runner):
        result = runner.invoke(cli, ["quality-check", "-i", "/no/file.csv"])
        assert result.exit_code != 0

    def test_analyze_requires_input(self, runner):
        result = runner.invoke(cli, ["analyze"])
        assert result.exit_code != 0


# ==================================================================
# quality-check on sample data
# ==================================================================

class TestQualityCheck:
    @pytest.mark.skipif(not _has_sample_data(), reason="Sample data not generated")
    def test_runs_on_sample_data(self, runner):
        result = runner.invoke(cli, [
            "quality-check", "-i", str(_INVOICES_CSV),
        ])
        assert result.exit_code == 0


# ==================================================================
# feedback
# ==================================================================

class TestFeedback:
    def test_records_feedback(self, runner, tmp_path):
        # Override DB path via env or just let it use default
        result = runner.invoke(cli, [
            "feedback",
            "--finding-id", "SA-test123",
            "--verdict", "false_positive",
            "--customer-id", "TEST",
            "--notes", "Not fraud",
        ])
        assert result.exit_code == 0
        assert "SA-test123" in result.output
        assert "false_positive" in result.output

    def test_invalid_verdict(self, runner):
        result = runner.invoke(cli, [
            "feedback",
            "--finding-id", "SA-test",
            "--verdict", "invalid_choice",
        ])
        assert result.exit_code != 0


# ==================================================================
# generate-sample-data
# ==================================================================

class TestGenerateSampleData:
    def test_generates_data(self, runner, tmp_path):
        out = str(tmp_path / "gen_data")
        result = runner.invoke(cli, [
            "generate-sample-data", "-o", out, "--seed", "99",
        ])
        assert result.exit_code == 0
        assert (Path(out) / "sample_invoices" / "invoices.csv").exists()


# ==================================================================
# analyze — full pipeline on sample data
# ==================================================================

class TestAnalyze:
    @pytest.mark.skipif(not _has_sample_data(), reason="Sample data not generated")
    def test_full_pipeline(self, runner, tmp_path):
        out_dir = str(tmp_path / "results")
        result = runner.invoke(cli, [
            "analyze",
            "-i", str(_INVOICES_CSV),
            "-v", str(_VENDORS_CSV),
            "-c", str(_CONTRACTS_JSON),
            "-a", str(_THRESHOLDS_JSON),
            "-r", str(_RECEIPTS_CSV),
            "-o", out_dir,
        ])
        assert result.exit_code == 0, result.output
        assert "Analysis Complete" in result.output
        assert (Path(out_dir) / "executive_summary.json").exists()
        assert (Path(out_dir) / "dashboard_data.json").exists()

    @pytest.mark.skipif(not _has_sample_data(), reason="Sample data not generated")
    def test_skip_modules(self, runner, tmp_path):
        out_dir = str(tmp_path / "results_skip")
        result = runner.invoke(cli, [
            "analyze",
            "-i", str(_INVOICES_CSV),
            "-o", out_dir,
            "--skip-modules", "vendor_collusion",
            "--skip-modules", "market_price",
        ])
        assert result.exit_code == 0
        assert "Skipping vendor_collusion" in result.output


# ==================================================================
# validate
# ==================================================================

class TestValidate:
    @pytest.mark.skipif(not _has_sample_data(), reason="Sample data not generated")
    def test_validation_report(self, runner, tmp_path):
        out_dir = str(tmp_path / "val")
        result = runner.invoke(cli, [
            "validate",
            "-i", str(_DATA_DIR),
            "-k", str(_FRAUD_KEY),
            "-o", out_dir,
        ])
        assert result.exit_code == 0
        assert "OVERALL" in result.output
        assert (Path(out_dir) / "validation_report.json").exists()
