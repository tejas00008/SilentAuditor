"""Tests for src.normalization.item_normalizer."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.normalization.cache import CacheManager
from src.normalization.item_normalizer import (
    ItemNormalizer,
    _build_keyword_map,
    DEFAULT_TAXONOMY,
)
from src.utils.constants import LineItem


@pytest.fixture
def cache(tmp_path):
    cm = CacheManager(db_path=str(tmp_path / "test.db"))
    yield cm
    cm.close()


@pytest.fixture
def mock_llm():
    """LLMClient mock that supports classify_by_keywords and classify_line_item."""
    from src.utils.llm_client import LLMClient
    mock = MagicMock(spec=LLMClient)
    # Default: classify_by_keywords delegates to the real implementation
    # so the normalizer's Tier 1 path works via _keyword_map.
    # We use side_effect to call the real static-ish method.
    def _kw_classify(desc, kw_map):
        if not desc or not kw_map:
            return None
        desc_lower = desc.lower()
        best_match = None
        best_len = 0
        for kw, cat in kw_map.items():
            if kw in desc_lower and len(kw) > best_len:
                best_match = cat
                best_len = len(kw)
        return best_match

    mock.classify_by_keywords.side_effect = _kw_classify
    mock.classify_line_item.return_value = {
        "category": "Uncategorized",
        "subcategory": None,
        "item_type": None,
        "confidence": 0.70,
        "source": "llm",
    }
    return mock


@pytest.fixture
def normalizer(cache, mock_llm):
    return ItemNormalizer(cache=cache, llm_client=mock_llm)


def _make_item(desc: str, item_id: str = "LI-001") -> LineItem:
    return LineItem(
        line_item_id=item_id,
        invoice_id="INV-001",
        description=desc,
        total_amount=Decimal("100.00"),
    )


# ==================================================================
# _build_keyword_map
# ==================================================================

class TestBuildKeywordMap:
    def test_contains_item_types(self):
        km = _build_keyword_map(DEFAULT_TAXONOMY)
        assert "steel rebar" in km
        assert "Ready Mix" not in km  # should be lowered
        assert "ready mix" in km

    def test_contains_subcategories(self):
        km = _build_keyword_map(DEFAULT_TAXONOMY)
        assert "consulting" in km

    def test_contains_single_words(self):
        km = _build_keyword_map(DEFAULT_TAXONOMY)
        assert "plumbing" in km
        assert "electrical" in km


# ==================================================================
# Tier 1: keyword matching
# ==================================================================

class TestTier1KeywordClassification:
    def test_steel_rebar(self, normalizer):
        result = normalizer.normalize_single("Steel rebar 40mm grade 60", "C1")
        assert result["category"] == "Raw Materials"
        assert result["source"] == "rule"
        assert result["confidence"] == 0.90

    def test_plumbing(self, normalizer):
        result = normalizer.normalize_single("Plumbing repair bathroom 2nd floor", "C1")
        assert "Construction" in result["category"]
        assert result["source"] == "rule"

    def test_janitorial(self, normalizer):
        result = normalizer.normalize_single("Janitorial services - March 2025", "C1")
        assert result["category"] == "Facilities"
        assert result["source"] == "rule"

    def test_software_licenses(self, normalizer):
        result = normalizer.normalize_single("Annual software licenses renewal", "C1")
        assert result["category"] == "Office & Administrative"

    def test_consulting(self, normalizer):
        result = normalizer.normalize_single("IT consulting for ERP migration", "C1")
        assert "Professional Services" in (result["category"] or "")

    def test_fuel(self, normalizer):
        result = normalizer.normalize_single("Diesel fuel 500 gallons", "C1")
        assert result["category"] == "Transportation"


# ==================================================================
# Tier 2/3: LLM fallback
# ==================================================================

class TestTier2And3:
    def test_falls_to_llm_for_unknown(self, normalizer, mock_llm):
        mock_llm.classify_line_item.return_value = {
            "category": "Specialized Equipment",
            "subcategory": "Lab",
            "item_type": "Centrifuge",
            "confidence": 0.80,
            "source": "llm",
        }
        result = normalizer.normalize_single(
            "Beckman Coulter Centrifuge model 5810R", "C1",
        )
        assert result["category"] == "Specialized Equipment"
        assert result["source"] == "llm"
        mock_llm.classify_line_item.assert_called_once()

    def test_embedding_source_tracked(self, normalizer, mock_llm):
        mock_llm.classify_line_item.return_value = {
            "category": "Construction",
            "subcategory": "Labor",
            "item_type": None,
            "confidence": 0.87,
            "source": "embedding",
        }
        # Use a description that won't match any keyword in the taxonomy
        result = normalizer.normalize_single(
            "Mobilization of crew for Phase 2 sitework", "C1",
        )
        assert result["source"] == "embedding"
        stats = normalizer.get_normalization_stats()
        assert stats["tier2_embedding"] == 1


# ==================================================================
# Cache behaviour
# ==================================================================

class TestCaching:
    def test_cache_hit_skips_classification(self, cache, mock_llm):
        # Pre-populate cache
        cache.set_item_classification(
            "test item", "test item", "CatA", "SubA",
            "TypeA", None, "rule", 0.95,
        )
        n = ItemNormalizer(cache=cache, llm_client=mock_llm)
        result = n.normalize_single("test item", "C1")
        assert result["category"] == "CatA"
        assert result["source"] == "rule"
        mock_llm.classify_by_keywords.assert_not_called()
        assert n.get_normalization_stats()["cache_hits"] == 1

    def test_new_classification_stored_in_cache(self, normalizer, cache):
        normalizer.normalize_single("Steel rebar 40mm", "C1")
        cached = cache.get_item_classification("Steel rebar 40mm")
        assert cached is not None
        assert cached["category"] == "Raw Materials"


# ==================================================================
# normalize_all
# ==================================================================

class TestNormalizeAll:
    def test_enriches_line_items(self, normalizer):
        items = [
            _make_item("Steel rebar 20mm", "LI-001"),
            _make_item("Plumbing repair", "LI-002"),
        ]
        result = normalizer.normalize_all(items, "C1")
        assert result[0].category == "Raw Materials"
        assert result[1].category is not None
        assert result[0].normalized_description is not None

    def test_empty_description(self, normalizer):
        items = [_make_item("")]
        result = normalizer.normalize_all(items, "C1")
        assert result[0].category is None


# ==================================================================
# Stats
# ==================================================================

class TestStats:
    def test_initial_zero(self, normalizer):
        stats = normalizer.get_normalization_stats()
        assert stats["total"] == 0
        assert stats["tier1_rule"] == 0

    def test_counts_accumulate(self, normalizer, mock_llm):
        normalizer.normalize_single("Steel rebar", "C1")  # tier 1
        mock_llm.classify_line_item.return_value = {
            "category": "X", "source": "llm", "confidence": 0.7,
        }
        normalizer.normalize_single("xyzzy unknown widget", "C1")  # tier 3
        stats = normalizer.get_normalization_stats()
        assert stats["tier1_rule"] == 1
        assert stats["tier3_llm"] == 1
        assert stats["total"] == 2
