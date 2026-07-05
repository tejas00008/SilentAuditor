"""Tests for src/utils/llm_client — tiered LLM processing client.

All Anthropic API calls and sentence-transformer model loads are mocked.
No real API calls are made.
"""

import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.normalization.cache import CacheManager
from src.utils.llm_client import (
    CostLimitExceeded,
    LLMClient,
    _estimate_cost,
    _try_parse_json,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    db_path = str(tmp_path / "test.db")
    cm = CacheManager(db_path=db_path)
    yield cm
    cm.close()


@pytest.fixture
def client(cache):
    """LLMClient with mocked embedding model (no network access)."""
    c = LLMClient(cache=cache, config={"llm_max_monthly_cost_per_customer": 200.0})
    # Pre-inject a mock embedding model so _get_embedding_model() is never called
    mock_model = MagicMock()
    mock_model.encode = MagicMock(
        side_effect=lambda text, **kw: (
            np.random.default_rng(hash(text) % 2**32).random(384).astype(np.float32)
            if isinstance(text, str)
            else np.array([
                np.random.default_rng(hash(t) % 2**32).random(384).astype(np.float32)
                for t in text
            ])
        )
    )
    c._embedding_model = mock_model
    return c


def _make_api_response(text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Build a fake Anthropic API response object."""
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

class TestEstimateCost:
    def test_basic(self):
        cost = _estimate_cost(1_000_000, 0)
        assert cost == pytest.approx(3.0)

    def test_output(self):
        cost = _estimate_cost(0, 1_000_000)
        assert cost == pytest.approx(15.0)

    def test_combined(self):
        cost = _estimate_cost(100, 50)
        assert cost == pytest.approx(100 * 3e-6 + 50 * 15e-6)

    def test_zero(self):
        assert _estimate_cost(0, 0) == 0.0


class TestTryParseJson:
    def test_valid_json(self):
        assert _try_parse_json('{"a": 1}') == {"a": 1}

    def test_json_with_whitespace(self):
        assert _try_parse_json('  {"a": 1}  ') == {"a": 1}

    def test_embedded_json(self):
        text = 'Here is the result: {"foo": "bar"} hope that helps'
        assert _try_parse_json(text) == {"foo": "bar"}

    def test_plain_text_fallback(self):
        result = _try_parse_json("not json at all")
        assert result == {"text": "not json at all"}

    def test_empty(self):
        result = _try_parse_json("")
        assert result == {"text": ""}


# ==================================================================
# TIER 1 — Rule-Based
# ==================================================================

class TestClassifyByKeywords:
    def test_match(self, client):
        kw = {"consulting": "Professional Services", "steel": "Raw Materials > Metals"}
        assert client.classify_by_keywords("Management consulting Q4", kw) == "Professional Services"

    def test_longest_match(self, client):
        kw = {"steel": "Metals", "stainless steel": "Metals > Stainless"}
        result = client.classify_by_keywords("Stainless steel rebar", kw)
        assert result == "Metals > Stainless"

    def test_no_match(self, client):
        kw = {"consulting": "PS"}
        assert client.classify_by_keywords("Grade A concrete mix", kw) is None

    def test_empty_description(self, client):
        assert client.classify_by_keywords("", {"a": "b"}) is None

    def test_empty_map(self, client):
        assert client.classify_by_keywords("hello", {}) is None

    def test_case_insensitive(self, client):
        kw = {"plumbing": "Trade Services"}
        assert client.classify_by_keywords("PLUMBING repair work", kw) == "Trade Services"


class TestScoreDescriptionVagueness:
    def test_very_vague(self, client):
        score = client.score_description_vagueness("Services")
        assert score < 20

    def test_generic_only(self, client):
        score = client.score_description_vagueness("miscellaneous fees")
        assert score < 30

    def test_specific(self, client):
        score = client.score_description_vagueness(
            "Installation of 200 sqft granite countertop on 01/15/2025"
        )
        assert score > 60

    def test_deliverable(self, client):
        score = client.score_description_vagueness(
            "Final inspection report for Building A north wing"
        )
        assert score > 50

    def test_empty(self, client):
        assert client.score_description_vagueness("") == 0

    def test_clamped_to_range(self, client):
        score = client.score_description_vagueness("x")
        assert 0 <= score <= 100


# ==================================================================
# TIER 2 — Embeddings
# ==================================================================

class TestGetEmbedding:
    def test_returns_vector(self, client):
        vec = client.get_embedding("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)

    def test_cache_hit(self, client):
        client.get_embedding("same text")
        client.get_embedding("same text")
        assert client._embedding_cache_hits >= 1

    def test_different_texts_differ(self, client):
        v1 = client.get_embedding("alpha")
        v2 = client.get_embedding("beta")
        assert not np.array_equal(v1, v2)


class TestGetBatchEmbeddings:
    def test_basic(self, client):
        texts = ["foo", "bar", "baz"]
        results = client.get_batch_embeddings(texts)
        assert len(results) == 3
        assert all(v.shape == (384,) for v in results)

    def test_partial_cache(self, client):
        # Pre-cache one (miss on first call, stores in cache)
        client.get_embedding("cached")
        assert client._embedding_cache_misses == 1
        results = client.get_batch_embeddings(["cached", "new"])
        assert len(results) == 2
        # "cached" should have been a hit in the batch call
        assert client._embedding_cache_hits >= 1
        # "new" should have been a miss
        assert client._embedding_cache_misses == 2


class TestComputeTextSimilarity:
    def test_same_text(self, client):
        sim = client.compute_text_similarity("hello", "hello")
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_range(self, client):
        sim = client.compute_text_similarity("cat", "dog")
        assert 0.0 <= sim <= 1.0


# ==================================================================
# TIER 3 — Claude API
# ==================================================================

class TestCallLLM:
    def test_cache_hit(self, client, cache):
        cache.set_llm_response("test prompt", '{"answer":"cached"}', "m", 10, 5, 0.001)
        result = client.call_llm("test prompt", "C1", "mod")
        assert result == {"answer": "cached"}
        assert client._llm_cache_hits == 1

    def test_cache_miss_calls_api(self, client):
        response = _make_api_response('{"answer":"fresh"}')
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        client._anthropic_client = mock_client

        result = client.call_llm("new prompt", "C1", "mod", use_cache=True)
        assert result == {"answer": "fresh"}
        mock_client.messages.create.assert_called_once()

    def test_cost_tracking(self, client):
        response = _make_api_response('{"ok":true}', input_tokens=200, output_tokens=100)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        client._anthropic_client = mock_client

        client.call_llm("prompt", "CUST1", "dup")
        assert client.get_session_cost() > Decimal("0")

    def test_cost_limit_exceeded(self, client, cache):
        # Burn through the budget
        cache.log_llm_cost("CUST1", "m", 0, 0, 200.01)
        with pytest.raises(CostLimitExceeded):
            mock_client = MagicMock()
            client._anthropic_client = mock_client
            client.call_llm("prompt", "CUST1", "mod")

    def test_cost_warning_logged(self, client, cache, caplog):
        # 80% of $200 = $160
        cache.log_llm_cost("CUST1", "m", 0, 0, 165.0)
        response = _make_api_response('{"ok":true}')
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        client._anthropic_client = mock_client

        import logging
        with caplog.at_level(logging.WARNING):
            client.call_llm("prompt", "CUST1", "mod")
        assert any("approaching" in r.message.lower() for r in caplog.records)

    def test_use_cache_false(self, client, cache):
        cache.set_llm_response("p", '{"old":"data"}', "m", 10, 5, 0.001)

        response = _make_api_response('{"new":"data"}')
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        client._anthropic_client = mock_client

        result = client.call_llm("p", "C1", "mod", use_cache=False)
        assert result == {"new": "data"}
        mock_client.messages.create.assert_called_once()

    def test_non_json_response(self, client):
        response = _make_api_response("I think it might be a duplicate")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        client._anthropic_client = mock_client

        result = client.call_llm("p", "C1", "m")
        assert result["text"] == "I think it might be a duplicate"

    def test_stores_in_cache(self, client, cache):
        response = _make_api_response('{"stored":true}')
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        client._anthropic_client = mock_client

        client.call_llm("cacheable prompt", "C1", "m")
        cached = cache.get_llm_response("cacheable prompt")
        assert cached == {"stored": True}


class TestCallWithRetry:
    def test_retry_on_rate_limit(self, client):
        import anthropic

        mock_client = MagicMock()
        err_response = SimpleNamespace(status_code=429, headers={})
        rate_err = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        ok_response = _make_api_response('{"ok":true}')
        mock_client.messages.create.side_effect = [rate_err, ok_response]
        client._anthropic_client = mock_client

        with patch("src.utils.llm_client.time.sleep"):
            result = client.call_llm("p", "C1", "m")
        assert result == {"ok": True}
        assert mock_client.messages.create.call_count == 2

    def test_retry_on_timeout(self, client):
        import anthropic

        mock_client = MagicMock()
        timeout_err = anthropic.APITimeoutError(request=MagicMock())
        ok_response = _make_api_response('{"ok":true}')
        mock_client.messages.create.side_effect = [timeout_err, ok_response]
        client._anthropic_client = mock_client

        with patch("src.utils.llm_client.time.sleep"):
            result = client.call_llm("p", "C1", "m")
        assert result == {"ok": True}

    def test_exhausted_retries(self, client):
        import anthropic

        mock_client = MagicMock()
        timeout_err = anthropic.APITimeoutError(request=MagicMock())
        mock_client.messages.create.side_effect = [timeout_err] * 3
        client._anthropic_client = mock_client

        with patch("src.utils.llm_client.time.sleep"):
            with pytest.raises(anthropic.APITimeoutError):
                client.call_llm("p", "C1", "m")


# ==================================================================
# Missing API key
# ==================================================================

class TestMissingAPIKey:
    def test_raises_on_missing_key(self, cache):
        c = LLMClient(cache=cache)
        c._embedding_model = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                c.call_llm("prompt", "C1", "m")


# ==================================================================
# High-level classification
# ==================================================================

class TestClassifyLineItem:
    def test_tier1_keyword(self, client):
        taxonomy = {"keywords": {"plumbing": "Trade Services"}}
        result = client.classify_line_item("Plumbing repair bathroom", taxonomy, "C1")
        assert result["category"] == "Trade Services"
        assert result["source"] == "rule"
        assert result["confidence"] == 0.90

    def test_tier2_embedding(self, client):
        # Build fake category embeddings that match perfectly
        desc = "concrete pour"
        desc_vec = client.get_embedding(desc)
        taxonomy = {
            "keywords": {},
            "category_embeddings": {"Construction Materials": desc_vec},
        }
        result = client.classify_line_item(desc, taxonomy, "C1")
        assert result["category"] == "Construction Materials"
        assert result["source"] == "embedding"

    def test_tier3_llm(self, client):
        response = _make_api_response(
            '{"category":"Electrical","subcategory":"Wiring","item_type":"service","confidence":0.88}'
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        client._anthropic_client = mock_client

        taxonomy = {"keywords": {}, "categories": ["Electrical", "Plumbing"]}
        result = client.classify_line_item("Rewire panel 200A", taxonomy, "C1")
        assert result["category"] == "Electrical"
        assert result["source"] == "llm"

    def test_fallback_defaults(self, client):
        response = _make_api_response('{"category":"Misc"}')
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        client._anthropic_client = mock_client

        taxonomy = {"keywords": {}, "categories": ["Misc"]}
        result = client.classify_line_item("xyz", taxonomy, "C1")
        assert result["subcategory"] is None
        assert result["source"] == "llm"


class TestClassifyLineItemsBatch:
    def test_mixed_tiers(self, client):
        response = _make_api_response(
            '[{"category":"Other","confidence":0.7}]'
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        client._anthropic_client = mock_client

        taxonomy = {
            "keywords": {"steel": "Metals"},
            "categories": ["Metals", "Other"],
        }
        results = client.classify_line_items_batch(
            ["steel rebar", "unknown widget"], taxonomy, "C1",
        )
        assert len(results) == 2
        assert results[0]["source"] == "rule"
        assert results[0]["category"] == "Metals"
        assert results[1]["source"] == "llm"


# ==================================================================
# Duplicate / scope assessment
# ==================================================================

class TestAssessDuplicatePair:
    def test_basic(self, client):
        response = _make_api_response(
            '{"is_duplicate":true,"confidence":0.92,"reasoning":"same amounts"}'
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        client._anthropic_client = mock_client

        inv_a = {"vendor_name": "Acme", "date": "2025-01-01", "amount": "1000",
                 "line_items": "Widget x10", "po_number": "PO-1"}
        inv_b = {"vendor_name": "Acme", "date": "2025-01-01", "amount": "1000",
                 "line_items": "Widget x10"}
        result = client.assess_duplicate_pair(inv_a, inv_b, "C1")
        assert result["is_duplicate"] is True
        assert result["confidence"] == 0.92

    def test_defaults_on_partial_response(self, client):
        response = _make_api_response('{"reasoning":"unclear"}')
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        client._anthropic_client = mock_client

        inv_a = {"vendor_name": "A", "date": "d", "amount": "1", "line_items": "x"}
        inv_b = {"vendor_name": "B", "date": "d", "amount": "1", "line_items": "x"}
        result = client.assess_duplicate_pair(inv_a, inv_b, "C1")
        assert result["is_duplicate"] is False
        assert result["confidence"] == 0.5


class TestAssessScopeBoundary:
    def test_basic(self, client):
        response = _make_api_response(
            '{"in_scope":false,"confidence":0.85,"closest_scope_item":"Plumbing",'
            '"reasoning":"Electrical not in scope"}'
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        client._anthropic_client = mock_client

        result = client.assess_scope_boundary(
            "Electrical panel upgrade", ["Plumbing", "HVAC"], "C1",
        )
        assert result["in_scope"] is False


# ==================================================================
# Cost management
# ==================================================================

class TestCostManagement:
    def test_session_cost_starts_zero(self, client):
        assert client.get_session_cost() == Decimal("0")

    def test_session_cost_accumulates(self, client):
        response = _make_api_response('{"ok":true}', 500, 200)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = response
        client._anthropic_client = mock_client

        client.call_llm("p1", "C1", "m")
        cost1 = client.get_session_cost()
        assert cost1 > Decimal("0")

        # Second different prompt
        client.call_llm("p2", "C1", "m")
        assert client.get_session_cost() > cost1

    def test_customer_monthly_cost(self, client, cache):
        cache.log_llm_cost("C1", "m", 100, 50, 0.05)
        assert client.get_customer_monthly_cost("C1") == Decimal("0.05")

    def test_cost_breakdown(self, client, cache):
        cache.log_llm_cost("C1", "dup", 100, 50, 0.01)
        cache.log_llm_cost("C1", "price", 200, 100, 0.02)
        breakdown = client.get_cost_breakdown("C1")
        assert "dup" in breakdown
        assert "price" in breakdown


# ==================================================================
# Cache hit rates
# ==================================================================

class TestCacheHitRates:
    def test_initial_zero(self, client):
        rates = client.get_cache_hit_rates()
        assert rates["embedding_hit_rate_pct"] == 0.0
        assert rates["llm_hit_rate_pct"] == 0.0

    def test_after_activity(self, client, cache):
        # Embedding: 1 miss + 1 hit
        client.get_embedding("test")
        client.get_embedding("test")
        rates = client.get_cache_hit_rates()
        assert rates["embedding_hits"] == 1
        assert rates["embedding_misses"] == 1
        assert rates["embedding_hit_rate_pct"] == 50.0

    def test_llm_hit_rate(self, client, cache):
        cache.set_llm_response("cached_prompt", '{"a":1}', "m", 10, 5, 0.001)
        client.call_llm("cached_prompt", "C1", "m")
        rates = client.get_cache_hit_rates()
        assert rates["llm_hits"] == 1
        assert rates["llm_hit_rate_pct"] == 100.0
