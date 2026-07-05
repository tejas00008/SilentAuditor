"""Tests for src.normalization.cache.CacheManager."""

import json
import os
import tempfile
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from src.normalization.cache import CacheManager, _hash_text


@pytest.fixture
def cache(tmp_path):
    """Create a CacheManager backed by a temporary database."""
    db_path = str(tmp_path / "test_cache.db")
    cm = CacheManager(db_path=db_path)
    yield cm
    cm.close()


# ------------------------------------------------------------------
# Table creation
# ------------------------------------------------------------------

class TestTableCreation:
    def test_tables_exist(self, cache):
        rows = cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = sorted(
            r["name"] for r in rows if not r["name"].startswith("sqlite_")
        )
        expected = sorted([
            "embedding_cache",
            "feedback",
            "field_mappings",
            "item_classifications",
            "llm_cache",
            "llm_cost_tracking",
            "vendor_aliases",
            "vendor_baselines",
        ])
        assert table_names == expected

    def test_wal_mode_enabled(self, cache):
        row = cache._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"

    def test_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "cache.db"
        cm = CacheManager(db_path=str(nested))
        assert nested.parent.exists()
        cm.close()


# ------------------------------------------------------------------
# Vendor alias operations
# ------------------------------------------------------------------

class TestVendorAliases:
    def test_set_and_get(self, cache):
        cache.set_vendor_alias("Acme Corp.", "V001", "Acme Corporation", 0.95)
        result = cache.get_canonical_vendor("Acme Corp.")
        assert result == ("V001", "Acme Corporation", 0.95)

    def test_missing_alias_returns_none(self, cache):
        assert cache.get_canonical_vendor("nonexistent") is None

    def test_upsert_overwrites(self, cache):
        cache.set_vendor_alias("acme", "V001", "Acme Corp", 0.80)
        cache.set_vendor_alias("acme", "V001", "Acme Corporation", 0.95)
        result = cache.get_canonical_vendor("acme")
        assert result[1] == "Acme Corporation"
        assert result[2] == 0.95

    def test_unicode_alias(self, cache):
        cache.set_vendor_alias("株式会社テスト", "V099", "Test KK", 0.90)
        result = cache.get_canonical_vendor("株式会社テスト")
        assert result == ("V099", "Test KK", 0.90)

    def test_get_all_canonical_vendors(self, cache):
        cache.set_vendor_alias("a1", "V001", "Acme", 0.9)
        cache.set_vendor_alias("a2", "V001", "Acme", 0.9)
        cache.set_vendor_alias("b1", "V002", "Beta Inc", 0.8)
        vendors = cache.get_all_canonical_vendors()
        assert ("V001", "Acme") in vendors
        assert ("V002", "Beta Inc") in vendors

    def test_verify_alias(self, cache):
        cache.set_vendor_alias("acme", "V001", "Acme", 0.8, verified=False)
        cache.verify_vendor_alias("acme")
        row = cache._conn.execute(
            "SELECT verified_by_human FROM vendor_aliases WHERE alias_text = 'acme'"
        ).fetchone()
        assert row["verified_by_human"] == 1

    def test_verified_flag_on_create(self, cache):
        cache.set_vendor_alias("acme", "V001", "Acme", 0.99, verified=True)
        row = cache._conn.execute(
            "SELECT verified_by_human FROM vendor_aliases WHERE alias_text = 'acme'"
        ).fetchone()
        assert row["verified_by_human"] == 1


# ------------------------------------------------------------------
# Item classification operations
# ------------------------------------------------------------------

class TestItemClassification:
    def test_set_and_get(self, cache):
        cache.set_item_classification(
            description="Concrete pour - 4000 PSI",
            normalized="concrete pour 4000 psi",
            category="Materials",
            subcategory="Concrete",
            item_type="commodity",
            unit="cubic_yard",
            source="rule",
            confidence=0.92,
        )
        result = cache.get_item_classification("Concrete pour - 4000 PSI")
        assert result is not None
        assert result["category"] == "Materials"
        assert result["subcategory"] == "Concrete"
        assert result["unit_of_measure"] == "cubic_yard"
        assert result["classification_source"] == "rule"

    def test_missing_returns_none(self, cache):
        assert cache.get_item_classification("nope") is None

    def test_upsert_updates_classification(self, cache):
        cache.set_item_classification("desc", "d", "CatA", None, None, None, "rule", 0.5)
        cache.set_item_classification("desc", "d", "CatB", "SubB", None, None, "llm", 0.95)
        result = cache.get_item_classification("desc")
        assert result["category"] == "CatB"
        assert result["classification_source"] == "llm"

    def test_unicode_description(self, cache):
        cache.set_item_classification(
            "Béton armé 30MPa", "beton arme 30mpa", "Matériaux", None, None, None, "rule", 0.8
        )
        result = cache.get_item_classification("Béton armé 30MPa")
        assert result["category"] == "Matériaux"

    def test_optional_fields_nullable(self, cache):
        cache.set_item_classification("widget", "widget", "Parts", None, None, None, "rule", 0.7)
        result = cache.get_item_classification("widget")
        assert result["subcategory"] is None
        assert result["item_type"] is None
        assert result["unit_of_measure"] is None


# ------------------------------------------------------------------
# Embedding operations
# ------------------------------------------------------------------

class TestEmbeddingCache:
    def test_set_and_get(self, cache):
        vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        cache.set_embedding("hello world", vec, "test-model")
        result = cache.get_embedding("hello world", "test-model")
        assert result is not None
        np.testing.assert_array_almost_equal(result, vec)

    def test_missing_returns_none(self, cache):
        assert cache.get_embedding("missing", "model") is None

    def test_different_models_different_keys(self, cache):
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        cache.set_embedding("text", v1, "model-a")
        cache.set_embedding("text", v2, "model-b")
        r1 = cache.get_embedding("text", "model-a")
        r2 = cache.get_embedding("text", "model-b")
        np.testing.assert_array_almost_equal(r1, v1)
        np.testing.assert_array_almost_equal(r2, v2)

    def test_large_embedding(self, cache):
        vec = np.random.rand(384).astype(np.float32)
        cache.set_embedding("large", vec, "all-MiniLM-L6-v2")
        result = cache.get_embedding("large", "all-MiniLM-L6-v2")
        np.testing.assert_array_almost_equal(result, vec)

    def test_returned_array_is_writable(self, cache):
        vec = np.array([1.0, 2.0], dtype=np.float32)
        cache.set_embedding("test", vec, "m")
        result = cache.get_embedding("test", "m")
        result[0] = 99.0  # should not raise


# ------------------------------------------------------------------
# LLM cache operations
# ------------------------------------------------------------------

class TestLLMCache:
    def test_set_and_get(self, cache):
        response = json.dumps({"answer": "42"})
        cache.set_llm_response("what is the meaning?", response, "claude-3", 100, 50, 0.003)
        result = cache.get_llm_response("what is the meaning?")
        assert result == {"answer": "42"}

    def test_missing_returns_none(self, cache):
        assert cache.get_llm_response("unknown prompt") is None

    def test_upsert_updates_response(self, cache):
        cache.set_llm_response("p", '{"v":1}', "m1", 10, 5, 0.001)
        cache.set_llm_response("p", '{"v":2}', "m2", 20, 10, 0.002)
        result = cache.get_llm_response("p")
        assert result == {"v": 2}


# ------------------------------------------------------------------
# Feedback operations
# ------------------------------------------------------------------

class TestFeedback:
    def test_record_and_query(self, cache):
        cache.record_feedback("SA-dup-001", "CUST1", "false_positive", "Was a recurring payment")
        results = cache.get_feedback_for_pattern("dup", "V001")
        assert len(results) >= 1
        assert results[0]["verdict"] == "false_positive"

    def test_upsert_feedback(self, cache):
        cache.record_feedback("SA-001", "C1", "false_positive", "oops")
        cache.record_feedback("SA-001", "C1", "confirmed_fraud", "actually real")
        row = cache._conn.execute(
            "SELECT verdict FROM feedback WHERE finding_id = 'SA-001'"
        ).fetchone()
        assert row["verdict"] == "confirmed_fraud"

    def test_false_positive_count(self, cache):
        cache.record_feedback("SA-dup-001", "C1", "false_positive", None)
        cache.record_feedback("SA-dup-002", "C1", "false_positive", None)
        cache.record_feedback("SA-dup-003", "C1", "confirmed_fraud", None)
        count = cache.get_false_positive_count("dup", "pattern1", "C1")
        assert count == 2

    def test_false_positive_count_no_customer(self, cache):
        cache.record_feedback("SA-dup-010", "C1", "false_positive", None)
        cache.record_feedback("SA-dup-011", "C2", "false_positive", None)
        count = cache.get_false_positive_count("dup", "pattern1")
        assert count == 2

    def test_feedback_with_none_notes(self, cache):
        cache.record_feedback("SA-x", "C1", "legitimate", None)
        row = cache._conn.execute(
            "SELECT notes FROM feedback WHERE finding_id = 'SA-x'"
        ).fetchone()
        assert row["notes"] is None


# ------------------------------------------------------------------
# Field mapping operations
# ------------------------------------------------------------------

class TestFieldMappings:
    def test_set_and_get(self, cache):
        mappings = {
            "Invoice #": {"target_field": "invoice_number", "confidence": 0.95, "verified": True},
            "Amount": {"target_field": "total_amount", "confidence": 0.88, "verified": False},
        }
        cache.set_field_mappings("CUST1", mappings)
        result = cache.get_field_mappings("CUST1")
        assert result["Invoice #"]["target_field"] == "invoice_number"
        assert result["Invoice #"]["verified"] is True
        assert result["Amount"]["confidence"] == 0.88

    def test_missing_customer_returns_empty(self, cache):
        assert cache.get_field_mappings("NONEXISTENT") == {}

    def test_upsert_field_mapping(self, cache):
        cache.set_field_mappings("C1", {
            "Col1": {"target_field": "vendor_id", "confidence": 0.6, "verified": False},
        })
        cache.set_field_mappings("C1", {
            "Col1": {"target_field": "vendor_name", "confidence": 0.9, "verified": True},
        })
        result = cache.get_field_mappings("C1")
        assert result["Col1"]["target_field"] == "vendor_name"


# ------------------------------------------------------------------
# Vendor baseline operations
# ------------------------------------------------------------------

class TestVendorBaselines:
    def test_set_and_get(self, cache):
        baseline = {
            "avg_amount": "1234.56",
            "std_amount": "200.00",
            "avg_frequency_days": 30,
        }
        cache.set_vendor_baseline("V001", "CUST1", baseline)
        result = cache.get_vendor_baseline("V001", "CUST1")
        assert result == baseline

    def test_missing_returns_none(self, cache):
        assert cache.get_vendor_baseline("nope", "nope") is None

    def test_upsert_baseline(self, cache):
        cache.set_vendor_baseline("V1", "C1", {"avg": 100})
        cache.set_vendor_baseline("V1", "C1", {"avg": 200})
        result = cache.get_vendor_baseline("V1", "C1")
        assert result["avg"] == 200

    def test_different_customers_separate(self, cache):
        cache.set_vendor_baseline("V1", "C1", {"avg": 100})
        cache.set_vendor_baseline("V1", "C2", {"avg": 999})
        assert cache.get_vendor_baseline("V1", "C1")["avg"] == 100
        assert cache.get_vendor_baseline("V1", "C2")["avg"] == 999


# ------------------------------------------------------------------
# Cost tracking
# ------------------------------------------------------------------

class TestCostTracking:
    def test_log_and_get_monthly(self, cache):
        cache.log_llm_cost("CUST1", "duplicate_detection", 500, 200, 0.01)
        cache.log_llm_cost("CUST1", "price_creep", 300, 100, 0.005)
        total = cache.get_monthly_cost("CUST1")
        assert total == Decimal("0.015")

    def test_monthly_cost_zero_for_unknown(self, cache):
        assert cache.get_monthly_cost("NOBODY") == Decimal("0")

    def test_cost_by_module(self, cache):
        cache.log_llm_cost("C1", "dup", 100, 50, 0.002)
        cache.log_llm_cost("C1", "dup", 200, 100, 0.004)
        cache.log_llm_cost("C1", "price", 150, 75, 0.003)
        by_mod = cache.get_cost_by_module("C1")
        assert by_mod["dup"]["cost_usd"] == Decimal("0.006")
        assert by_mod["dup"]["input_tokens"] == 300
        assert by_mod["price"]["cost_usd"] == Decimal("0.003")

    def test_cost_by_module_empty(self, cache):
        assert cache.get_cost_by_module("NOBODY") == {}


# ------------------------------------------------------------------
# Maintenance
# ------------------------------------------------------------------

class TestMaintenance:
    def test_vacuum(self, cache):
        cache.vacuum()  # should not raise

    def test_cache_stats_empty(self, cache):
        stats = cache.get_cache_stats()
        assert stats["total_rows"] == 0
        assert all(v == 0 for v in stats["tables"].values())
        assert stats["database_size_bytes"] > 0

    def test_cache_stats_with_data(self, cache):
        cache.set_vendor_alias("a", "V1", "A", 0.9)
        cache.set_vendor_alias("b", "V2", "B", 0.8)
        cache.set_item_classification("x", "x", "Cat", None, None, None, "rule", 0.7)
        stats = cache.get_cache_stats()
        assert stats["tables"]["vendor_aliases"] == 2
        assert stats["tables"]["item_classifications"] == 1
        assert stats["total_rows"] == 3

    def test_close(self, tmp_path):
        cm = CacheManager(db_path=str(tmp_path / "close_test.db"))
        cm.close()
        # Connection should be closed — operations should fail
        with pytest.raises(Exception):
            cm._conn.execute("SELECT 1")


# ------------------------------------------------------------------
# Hash utility
# ------------------------------------------------------------------

class TestHashText:
    def test_deterministic(self):
        assert _hash_text("hello") == _hash_text("hello")

    def test_different_inputs(self):
        assert _hash_text("a") != _hash_text("b")

    def test_unicode(self):
        h = _hash_text("日本語テスト")
        assert len(h) == 64  # SHA-256 hex digest length
