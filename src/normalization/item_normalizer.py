"""Line item description normalization into a standardized taxonomy.

Uses the LLMClient's tiered pipeline (rule → embedding → LLM) to
classify each line-item description into a three-level taxonomy:
Category > Subcategory > Item Type.

All results are cached so the same description is never classified twice.
"""

import logging
from typing import Optional

from src.normalization.cache import CacheManager
from src.utils.constants import LineItem
from src.utils.similarity import normalize_text

logger = logging.getLogger(__name__)

DEFAULT_TAXONOMY: dict = {
    "Professional Services": {
        "Consulting": [
            "Management Consulting", "IT Consulting",
            "Financial Consulting", "Legal Consulting",
            "Strategy Consulting",
        ],
        "Accounting": [
            "Bookkeeping", "Tax Preparation",
            "Audit Services", "Payroll Services",
        ],
        "Legal": [
            "Legal Services", "Contract Review",
            "Compliance Advisory",
        ],
        "Engineering": [
            "Civil Engineering", "Structural Engineering",
            "MEP Engineering",
        ],
        "Architecture": [
            "Architectural Design", "Space Planning",
            "Interior Design",
        ],
    },
    "Construction": {
        "Labor": [
            "General Labor", "Skilled Labor",
            "Supervision", "Project Management",
        ],
        "Subcontracting": [
            "Electrical", "Plumbing", "HVAC", "Concrete",
            "Framing", "Roofing", "Painting", "Flooring",
        ],
        "Equipment Rental": [
            "Heavy Equipment", "Small Tools",
            "Scaffolding", "Temporary Facilities",
        ],
    },
    "Raw Materials": {
        "Metals": [
            "Steel Rebar", "Structural Steel", "Aluminum",
            "Copper", "Sheet Metal",
        ],
        "Concrete": [
            "Ready Mix", "Precast", "Cement", "Aggregates",
        ],
        "Lumber": [
            "Framing Lumber", "Plywood",
            "Engineered Wood", "Hardwood",
        ],
        "Pipes & Fittings": [
            "PVC Pipe", "Copper Pipe", "Steel Pipe",
            "Fittings", "Valves",
        ],
    },
    "Office & Administrative": {
        "Supplies": [
            "General Office Supplies", "Paper",
            "Toner/Ink", "Stationery",
        ],
        "Technology": [
            "Software Licenses", "Hardware",
            "IT Support", "Cloud Services",
        ],
        "Furniture": [
            "Office Furniture", "Ergonomic Equipment",
        ],
    },
    "Facilities": {
        "Maintenance": [
            "Building Maintenance", "Grounds Maintenance",
            "HVAC Maintenance",
        ],
        "Cleaning": [
            "Janitorial Services", "Window Cleaning",
            "Waste Management",
        ],
        "Security": [
            "Security Guards", "Security Systems",
            "Access Control",
        ],
        "Utilities": [
            "Electricity", "Water", "Gas",
            "Internet/Telecom",
        ],
    },
    "Healthcare": {
        "Medical Supplies": [
            "Disposables", "Instruments", "PPE",
            "Lab Supplies",
        ],
        "Pharmaceuticals": [
            "Medications", "Vaccines", "IV Solutions",
        ],
        "Equipment": [
            "Medical Equipment", "Diagnostic Equipment",
            "Patient Monitoring",
        ],
        "Services": [
            "Lab Services", "Imaging Services",
            "Staffing/Temporary",
        ],
    },
    "Transportation": {
        "Freight": [
            "Trucking", "Rail", "Air Freight",
            "Ocean Freight",
        ],
        "Courier": [
            "Express Delivery", "Local Delivery",
            "Mail Services",
        ],
        "Fleet": [
            "Vehicle Maintenance", "Fuel", "Leasing",
        ],
    },
    "Insurance & Finance": {
        "Insurance": [
            "General Liability", "Workers Comp",
            "Property Insurance", "Vehicle Insurance",
        ],
        "Financial": [
            "Banking Fees", "Factoring",
            "Credit Card Processing",
        ],
    },
}


def _build_keyword_map(taxonomy: dict) -> dict[str, str]:
    """Build a flat ``{keyword: "Category > Subcategory > Item Type"}`` map.

    Each item-type name is split into individual words and each word
    (3+ chars, lowered) becomes a keyword pointing to the full category
    path.  Longer, more specific multi-word phrases are also included so
    that the "longest-match" rule in ``classify_by_keywords`` prefers
    them.
    """
    kw_map: dict[str, str] = {}
    for category, subcats in taxonomy.items():
        for subcat, item_types in subcats.items():
            # Subcategory as keyword
            sub_lower = subcat.lower()
            path_sub = f"{category} > {subcat}"
            if len(sub_lower) >= 3:
                kw_map[sub_lower] = path_sub

            for item_type in item_types:
                full_path = f"{category} > {subcat} > {item_type}"
                it_lower = item_type.lower()
                # Full item-type phrase (most specific)
                kw_map[it_lower] = full_path
                # Individual words (less specific)
                for word in it_lower.split():
                    word = word.strip("/&")
                    if len(word) >= 3 and word not in ("and", "the", "for"):
                        kw_map.setdefault(word, full_path)
    return kw_map


class ItemNormalizer:
    """Classify line-item descriptions into a three-level taxonomy."""

    DEFAULT_TAXONOMY = DEFAULT_TAXONOMY

    def __init__(
        self,
        cache: CacheManager,
        llm_client: object,
        taxonomy: Optional[dict] = None,
    ) -> None:
        self.cache = cache
        self.llm_client = llm_client
        self.taxonomy = taxonomy or self.DEFAULT_TAXONOMY
        self._keyword_map: dict[str, str] = _build_keyword_map(self.taxonomy)

        # Counters
        self._tier1_count: int = 0
        self._tier2_count: int = 0
        self._tier3_count: int = 0
        self._cache_hit_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_all(
        self, line_items: list[LineItem], customer_id: str,
    ) -> list[LineItem]:
        """Normalize every line item, enriching each with category info."""
        for item in line_items:
            classification = self._classify(item.description, customer_id)
            item.normalized_description = classification.get("normalized_description")
            item.category = classification.get("category")
            item.subcategory = classification.get("subcategory")
            item.item_type = classification.get("item_type")
        return line_items

    def normalize_single(self, description: str, customer_id: str) -> dict:
        """Classify a single description string."""
        return self._classify(description, customer_id)

    def get_normalization_stats(self) -> dict:
        """Per-tier classification counts."""
        total = self._tier1_count + self._tier2_count + self._tier3_count + self._cache_hit_count
        return {
            "total": total,
            "cache_hits": self._cache_hit_count,
            "tier1_rule": self._tier1_count,
            "tier2_embedding": self._tier2_count,
            "tier3_llm": self._tier3_count,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _classify(self, description: str, customer_id: str) -> dict:
        """Tiered classification: cache → rule → embedding → LLM."""
        if not description or not description.strip():
            return self._empty_result(description)

        # ---- cache check -----------------------------------------------
        cached = self.cache.get_item_classification(description)
        if cached is not None:
            self._cache_hit_count += 1
            return {
                "normalized_description": cached.get("normalized_description"),
                "category": cached.get("category"),
                "subcategory": cached.get("subcategory"),
                "item_type": cached.get("item_type"),
                "confidence": cached.get("confidence"),
                "source": cached.get("classification_source"),
            }

        # ---- Tier 1: keyword classification ----------------------------
        norm_desc = normalize_text(description)
        cat_path = self.llm_client.classify_by_keywords(description, self._keyword_map)
        if cat_path is not None:
            parts = [p.strip() for p in cat_path.split(">")]
            result = self._parts_to_result(
                norm_desc, parts, confidence=0.90, source="rule",
            )
            self._store(description, result)
            self._tier1_count += 1
            return result

        # ---- Tier 2 / 3 via llm_client --------------------------------
        # Build taxonomy for the LLM client's classify_line_item
        categories_list = list(self.taxonomy.keys())
        taxonomy_payload = {
            "keywords": self._keyword_map,
            "categories": categories_list,
        }
        llm_result = self.llm_client.classify_line_item(
            description, taxonomy_payload, customer_id,
        )
        source = llm_result.get("source", "llm")
        if source == "embedding":
            self._tier2_count += 1
        else:
            self._tier3_count += 1

        result = {
            "normalized_description": norm_desc,
            "category": llm_result.get("category"),
            "subcategory": llm_result.get("subcategory"),
            "item_type": llm_result.get("item_type"),
            "confidence": llm_result.get("confidence", 0.70),
            "source": source,
        }
        self._store(description, result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parts_to_result(
        norm_desc: str, parts: list[str], confidence: float, source: str,
    ) -> dict:
        return {
            "normalized_description": norm_desc,
            "category": parts[0] if len(parts) >= 1 else None,
            "subcategory": parts[1] if len(parts) >= 2 else None,
            "item_type": parts[2] if len(parts) >= 3 else None,
            "confidence": confidence,
            "source": source,
        }

    @staticmethod
    def _empty_result(description: Optional[str]) -> dict:
        return {
            "normalized_description": None,
            "category": None,
            "subcategory": None,
            "item_type": None,
            "confidence": 0.0,
            "source": "rule",
        }

    def _store(self, description: str, result: dict) -> None:
        self.cache.set_item_classification(
            description=description,
            normalized=result.get("normalized_description", ""),
            category=result.get("category", ""),
            subcategory=result.get("subcategory"),
            item_type=result.get("item_type"),
            unit=None,
            source=result.get("source", "rule"),
            confidence=result.get("confidence", 0.0),
        )
