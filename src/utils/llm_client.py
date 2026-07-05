"""Tiered LLM processing client for SilentAuditor.

Manages a three-tier AI processing pipeline to control costs:

* **Tier 1 — Rule-based (zero cost):** keyword matching, regex, heuristics.
* **Tier 2 — Local embeddings (minimal cost):** sentence-transformers running
  locally with SQLite-cached vectors.
* **Tier 3 — Claude API (expensive):** only for ambiguous cases that the first
  two tiers cannot resolve.  Always cached, batched, and cost-tracked.

Every module that needs AI assistance goes through this client.
"""

import json
import logging
import os
import re
import time
from decimal import Decimal
from typing import Any, Optional

import numpy as np

from config import settings
from src.normalization.cache import CacheManager
from src.utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)

# Approximate per-token pricing for Claude Sonnet (USD per token).
_INPUT_COST_PER_TOKEN: float = 3.0 / 1_000_000   # $3 / MTok
_OUTPUT_COST_PER_TOKEN: float = 15.0 / 1_000_000  # $15 / MTok

_COST_WARNING_THRESHOLD: float = 0.80  # warn at 80% of monthly limit

_GENERIC_TERMS: set[str] = {
    "services", "service", "support", "consulting", "miscellaneous",
    "other", "fees", "fee", "charges", "charge", "expenses", "expense",
    "general", "various", "sundry", "professional",
}

_UNIT_WORDS: set[str] = {
    "hours", "hour", "hrs", "hr", "units", "unit", "sqft", "sq ft",
    "lbs", "lb", "kg", "tons", "ton", "gallons", "gal", "liters",
    "metres", "meters", "feet", "ft", "yards", "yd", "each", "ea",
    "pieces", "pcs", "boxes", "pallets", "rolls", "sheets", "bags",
}

_DELIVERABLE_WORDS: set[str] = {
    "report", "reports", "installation", "install", "delivery",
    "delivered", "repair", "repairs", "inspection", "audit",
    "assessment", "training", "maintenance", "deployment",
    "commissioning", "fabrication", "testing", "survey",
}

_MAX_RETRIES: int = 3
_RETRY_BASE_DELAY: float = 2.0  # seconds — doubled on each retry


class CostLimitExceeded(Exception):
    """Raised when a customer's monthly LLM cost limit has been reached."""
    pass


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from token counts."""
    return (input_tokens * _INPUT_COST_PER_TOKEN
            + output_tokens * _OUTPUT_COST_PER_TOKEN)


def _try_parse_json(text: str) -> dict:
    """Attempt to parse JSON from *text*, falling back to ``{"text": text}``."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try extracting a JSON object embedded in surrounding text.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {"text": text}


class LLMClient:
    """Tiered LLM processing client with caching and cost management."""

    def __init__(self, cache: CacheManager, config: Optional[dict] = None) -> None:
        self._cache = cache
        self._config = config or {}

        # Cost limits
        self._monthly_limit: float = self._config.get(
            "llm_max_monthly_cost_per_customer",
            settings.LLM_MAX_MONTHLY_COST_PER_CUSTOMER,
        )
        self._model: str = self._config.get("llm_model", settings.LLM_MODEL)
        self._embedding_model_name: str = self._config.get(
            "embedding_model", settings.EMBEDDING_MODEL,
        )

        # Lazy-initialised resources
        self._anthropic_client: Any = None
        self._embedding_model: Any = None

        # Session-level cost tracking
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        self._session_cost: Decimal = Decimal("0")

        # Cache hit/miss counters
        self._embedding_cache_hits: int = 0
        self._embedding_cache_misses: int = 0
        self._llm_cache_hits: int = 0
        self._llm_cache_misses: int = 0

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_anthropic_client(self) -> Any:
        """Return (and lazily create) the Anthropic client."""
        if self._anthropic_client is not None:
            return self._anthropic_client

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it before making LLM calls."
            )

        import anthropic
        self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self._anthropic_client

    def _get_embedding_model(self) -> Any:
        """Return (and lazily load) the sentence-transformer model."""
        if self._embedding_model is not None:
            return self._embedding_model

        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", self._embedding_model_name)
        self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    # ==================================================================
    # TIER 1 — Rule-Based (Zero Cost)
    # ==================================================================

    def classify_by_keywords(
        self, description: str, keyword_map: dict
    ) -> Optional[str]:
        """Match *description* against a keyword → category mapping.

        Returns the category for the longest matching keyword, or ``None``.
        """
        if not description or not keyword_map:
            return None

        desc_lower = description.lower()
        best_match: Optional[str] = None
        best_length: int = 0

        for keyword, category in keyword_map.items():
            kw_lower = keyword.lower()
            if kw_lower in desc_lower and len(kw_lower) > best_length:
                best_match = category
                best_length = len(kw_lower)

        return best_match

    def score_description_vagueness(self, description: str) -> int:
        """Rule-based specificity score for a line-item description.

        Returns an integer in ``[0, 100]`` — higher means **more specific**.
        """
        if not description:
            return 0

        score: int = 50
        words = description.split()
        word_count = len(words)

        # Penalise short descriptions
        if word_count < 3:
            score -= 30
        elif word_count < 5:
            score -= 15

        # Penalise generic-only text
        lower_words = {w.lower().strip(".,;:") for w in words}
        if lower_words and lower_words <= _GENERIC_TERMS:
            score -= 40

        desc_lower = description.lower()

        # Reward specifics
        if re.search(r"\d+\s*(?:" + "|".join(_UNIT_WORDS) + r")\b", desc_lower):
            score += 20

        if re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", description):
            score += 15

        # Proper nouns (capitalised words not at sentence start)
        if any(w[0].isupper() for w in words[1:] if w and w[0].isalpha()):
            score += 10

        # Unit measurements mentioned (without an adjacent digit)
        if any(u in desc_lower.split() for u in _UNIT_WORDS):
            score += 15

        # Deliverable keywords
        if any(d in desc_lower for d in _DELIVERABLE_WORDS):
            score += 15

        return max(0, min(100, score))

    # ==================================================================
    # TIER 2 — Local Embeddings (Minimal Cost)
    # ==================================================================

    def get_embedding(self, text: str) -> np.ndarray:
        """Get a cached or freshly computed embedding for *text*."""
        cached = self._cache.get_embedding(text, self._embedding_model_name)
        if cached is not None:
            self._embedding_cache_hits += 1
            return cached

        self._embedding_cache_misses += 1
        model = self._get_embedding_model()
        vec = model.encode(text, convert_to_numpy=True).astype(np.float32)
        self._cache.set_embedding(text, vec, self._embedding_model_name)
        return vec

    def get_batch_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Efficiently compute embeddings for a list of texts.

        Checks the cache for each text first and only computes the
        uncached ones in a single batch call.
        """
        results: list[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get_embedding(text, self._embedding_model_name)
            if cached is not None:
                results[i] = cached
                self._embedding_cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                self._embedding_cache_misses += 1

        if uncached_texts:
            model = self._get_embedding_model()
            vecs = model.encode(uncached_texts, convert_to_numpy=True, batch_size=32)
            for idx, text, vec in zip(uncached_indices, uncached_texts, vecs):
                vec32 = vec.astype(np.float32)
                self._cache.set_embedding(text, vec32, self._embedding_model_name)
                results[idx] = vec32

        return results  # type: ignore[return-value]

    def compute_text_similarity(self, text_a: str, text_b: str) -> float:
        """Semantic similarity (0–1) between two texts using cached embeddings."""
        vec_a = self.get_embedding(text_a)
        vec_b = self.get_embedding(text_b)
        sim = cosine_similarity(vec_a, vec_b)
        return max(0.0, min(1.0, sim))

    # ==================================================================
    # TIER 3 — Claude API (Expensive)
    # ==================================================================

    def call_llm(
        self,
        prompt: str,
        customer_id: str,
        module: str,
        use_cache: bool = True,
    ) -> dict:
        """Call Claude API with caching and cost tracking.

        Returns the parsed JSON response, or ``{"text": raw}`` when the
        response is not valid JSON.

        Raises:
            CostLimitExceeded: If the customer's monthly budget has been
                exceeded.
            RuntimeError: If ``ANTHROPIC_API_KEY`` is not set.
        """
        # 1. Cache lookup
        if use_cache:
            cached = self._cache.get_llm_response(prompt)
            if cached is not None:
                self._llm_cache_hits += 1
                logger.debug("LLM cache hit for module=%s", module)
                return cached
        self._llm_cache_misses += 1

        # 2. Budget check
        monthly_cost = self._cache.get_monthly_cost(customer_id)
        limit = Decimal(str(self._monthly_limit))

        if monthly_cost >= limit:
            raise CostLimitExceeded(
                f"Customer {customer_id} has exceeded the monthly LLM "
                f"budget (${monthly_cost} / ${limit})."
            )
        if monthly_cost >= limit * Decimal(str(_COST_WARNING_THRESHOLD)):
            logger.warning(
                "Customer %s approaching LLM cost limit: $%s / $%s",
                customer_id, monthly_cost, limit,
            )

        # 3. API call with retry
        client = self._get_anthropic_client()
        response = self._call_with_retry(client, prompt)

        # 4. Extract usage & cost
        input_tokens: int = response.usage.input_tokens
        output_tokens: int = response.usage.output_tokens
        cost = _estimate_cost(input_tokens, output_tokens)

        # 5. Parse response text
        raw_text: str = response.content[0].text
        parsed = _try_parse_json(raw_text)

        # 6. Persist
        response_json_str = json.dumps(parsed)
        self._cache.set_llm_response(
            prompt, response_json_str, self._model,
            input_tokens, output_tokens, cost,
        )
        self._cache.log_llm_cost(customer_id, module, input_tokens, output_tokens, cost)

        # 7. Session tracking
        self._session_input_tokens += input_tokens
        self._session_output_tokens += output_tokens
        self._session_cost += Decimal(str(cost))

        return parsed

    def _call_with_retry(self, client: Any, prompt: str) -> Any:
        """Call the Anthropic API with exponential back-off on retries."""
        import anthropic

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                return client.messages.create(
                    model=self._model,
                    max_tokens=1000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                )
            except (anthropic.RateLimitError, anthropic.APITimeoutError) as exc:
                last_exc = exc
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "API call attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES, type(exc).__name__, delay,
                )
                time.sleep(delay)
            except anthropic.APIError as exc:
                # Non-retryable API errors
                logger.error("Non-retryable API error: %s", exc)
                raise
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # High-level LLM helpers
    # ------------------------------------------------------------------

    def classify_line_item(
        self,
        description: str,
        taxonomy: dict,
        customer_id: str,
    ) -> dict:
        """Classify a line item through the tiered pipeline.

        Returns ``{category, subcategory, item_type, confidence, source}``.
        """
        # --- Tier 1: keyword classification --------------------------------
        keywords = taxonomy.get("keywords", {})
        cat = self.classify_by_keywords(description, keywords)
        if cat is not None:
            return {
                "category": cat,
                "subcategory": None,
                "item_type": None,
                "confidence": 0.90,
                "source": "rule",
            }

        # --- Tier 2: embedding similarity ----------------------------------
        category_embeddings = taxonomy.get("category_embeddings")
        if category_embeddings:
            desc_vec = self.get_embedding(description)
            best_score: float = 0.0
            best_cat: Optional[str] = None
            for cat_name, cat_vec in category_embeddings.items():
                sim = cosine_similarity(desc_vec, cat_vec)
                if sim > best_score:
                    best_score = sim
                    best_cat = cat_name
            if best_score >= 0.85 and best_cat is not None:
                return {
                    "category": best_cat,
                    "subcategory": None,
                    "item_type": None,
                    "confidence": round(best_score, 2),
                    "source": "embedding",
                }

        # --- Tier 3: LLM classification ------------------------------------
        categories_list = taxonomy.get("categories", [])
        prompt = (
            "You are an AP auditor classifying invoice line items.\n\n"
            f"Line item description: \"{description}\"\n\n"
            f"Available categories: {json.dumps(categories_list)}\n\n"
            "Respond ONLY with JSON: "
            '{"category": "...", "subcategory": "...", "item_type": "...", '
            '"confidence": 0.0-1.0}'
        )
        result = self.call_llm(prompt, customer_id, "classification")
        result["source"] = "llm"
        result.setdefault("category", "Uncategorized")
        result.setdefault("subcategory", None)
        result.setdefault("item_type", None)
        result.setdefault("confidence", 0.70)
        return result

    def classify_line_items_batch(
        self,
        descriptions: list[str],
        taxonomy: dict,
        customer_id: str,
    ) -> list[dict]:
        """Batch-classify line items, optimising LLM calls.

        Items resolved at Tier 1 or 2 skip the API entirely.  Remaining
        items are batched into groups of up to 10 for a single prompt each.
        """
        results: list[Optional[dict]] = [None] * len(descriptions)
        llm_pending: list[tuple[int, str]] = []

        keywords = taxonomy.get("keywords", {})
        category_embeddings = taxonomy.get("category_embeddings")

        for i, desc in enumerate(descriptions):
            # Tier 1
            cat = self.classify_by_keywords(desc, keywords)
            if cat is not None:
                results[i] = {
                    "category": cat, "subcategory": None,
                    "item_type": None, "confidence": 0.90, "source": "rule",
                }
                continue

            # Tier 2
            if category_embeddings:
                desc_vec = self.get_embedding(desc)
                best_score = 0.0
                best_cat = None
                for cat_name, cat_vec in category_embeddings.items():
                    sim = cosine_similarity(desc_vec, cat_vec)
                    if sim > best_score:
                        best_score = sim
                        best_cat = cat_name
                if best_score >= 0.85 and best_cat is not None:
                    results[i] = {
                        "category": best_cat, "subcategory": None,
                        "item_type": None, "confidence": round(best_score, 2),
                        "source": "embedding",
                    }
                    continue

            llm_pending.append((i, desc))

        # Tier 3 — batch LLM calls (groups of 10)
        categories_list = taxonomy.get("categories", [])
        batch_size = 10
        for batch_start in range(0, len(llm_pending), batch_size):
            batch = llm_pending[batch_start:batch_start + batch_size]
            numbered = "\n".join(
                f"{n+1}. \"{desc}\"" for n, (_, desc) in enumerate(batch)
            )
            prompt = (
                "You are an AP auditor classifying invoice line items.\n\n"
                f"Line items:\n{numbered}\n\n"
                f"Available categories: {json.dumps(categories_list)}\n\n"
                "Respond ONLY with a JSON array of objects, one per item "
                "in the same order: "
                '[{"category": "...", "subcategory": "...", "item_type": "...", '
                '"confidence": 0.0-1.0}, ...]'
            )
            response = self.call_llm(prompt, customer_id, "classification")

            # Parse array response
            llm_results: list[dict] = []
            if isinstance(response, list):
                llm_results = response
            elif isinstance(response, dict) and "text" not in response:
                llm_results = [response]
            else:
                raw = response.get("text", "")
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        llm_results = parsed
                except (json.JSONDecodeError, ValueError):
                    pass

            for j, (orig_idx, _) in enumerate(batch):
                if j < len(llm_results):
                    item = llm_results[j]
                    item["source"] = "llm"
                    item.setdefault("category", "Uncategorized")
                    item.setdefault("subcategory", None)
                    item.setdefault("item_type", None)
                    item.setdefault("confidence", 0.70)
                    results[orig_idx] = item
                else:
                    results[orig_idx] = {
                        "category": "Uncategorized", "subcategory": None,
                        "item_type": None, "confidence": 0.50, "source": "llm",
                    }

        return results  # type: ignore[return-value]

    def assess_duplicate_pair(
        self,
        invoice_a: dict,
        invoice_b: dict,
        customer_id: str,
    ) -> dict:
        """Ask Claude whether two invoices are likely duplicates.

        Returns ``{is_duplicate, confidence, reasoning}``.
        """
        prompt = (
            "You are an experienced AP auditor. Compare these two invoices "
            "and determine if they likely represent the same transaction "
            "billed twice, or two legitimate separate transactions.\n\n"
            f"Invoice A:\n"
            f"- Vendor: {invoice_a['vendor_name']}\n"
            f"- Date: {invoice_a['date']}\n"
            f"- Amount: ${invoice_a['amount']}\n"
            f"- Line items: {invoice_a['line_items']}\n"
            f"- PO Number: {invoice_a.get('po_number', 'N/A')}\n\n"
            f"Invoice B:\n"
            f"- Vendor: {invoice_b['vendor_name']}\n"
            f"- Date: {invoice_b['date']}\n"
            f"- Amount: ${invoice_b['amount']}\n"
            f"- Line items: {invoice_b['line_items']}\n"
            f"- PO Number: {invoice_b.get('po_number', 'N/A')}\n\n"
            'Respond ONLY with JSON: '
            '{"is_duplicate": true/false, "confidence": 0.0-1.0, '
            '"reasoning": "explanation"}'
        )
        result = self.call_llm(prompt, customer_id, "duplicate_detection")
        result.setdefault("is_duplicate", False)
        result.setdefault("confidence", 0.5)
        result.setdefault("reasoning", "")
        return result

    def assess_scope_boundary(
        self,
        line_item_desc: str,
        scope_of_work: list[str],
        customer_id: str,
    ) -> dict:
        """Ask Claude whether a line item falls within contract scope.

        Returns ``{in_scope, confidence, closest_scope_item, reasoning}``.
        """
        scope_str = "\n".join(f"- {s}" for s in scope_of_work)
        prompt = (
            "You are an experienced AP auditor reviewing contract compliance.\n\n"
            f"Line item charged: \"{line_item_desc}\"\n\n"
            f"Contract scope of work:\n{scope_str}\n\n"
            "Determine if this line item falls within the defined scope.\n\n"
            'Respond ONLY with JSON: '
            '{"in_scope": true/false, "confidence": 0.0-1.0, '
            '"closest_scope_item": "...", "reasoning": "explanation"}'
        )
        result = self.call_llm(prompt, customer_id, "contract_compliance")
        result.setdefault("in_scope", True)
        result.setdefault("confidence", 0.5)
        result.setdefault("closest_scope_item", "")
        result.setdefault("reasoning", "")
        return result

    # ==================================================================
    # Cost Management
    # ==================================================================

    def get_session_cost(self) -> Decimal:
        """Total LLM cost for the current in-memory session."""
        return self._session_cost

    def get_customer_monthly_cost(self, customer_id: str) -> Decimal:
        """Total LLM cost for *customer_id* during the current month."""
        return self._cache.get_monthly_cost(customer_id)

    def get_cost_breakdown(self, customer_id: str) -> dict:
        """Per-module cost breakdown for *customer_id* this month."""
        return self._cache.get_cost_by_module(customer_id)

    # ==================================================================
    # Cache Stats
    # ==================================================================

    def get_cache_hit_rates(self) -> dict:
        """Return hit-rate percentages for embeddings and LLM calls."""
        def _rate(hits: int, misses: int) -> float:
            total = hits + misses
            return round(hits / total * 100, 1) if total else 0.0

        return {
            "embedding_hits": self._embedding_cache_hits,
            "embedding_misses": self._embedding_cache_misses,
            "embedding_hit_rate_pct": _rate(
                self._embedding_cache_hits, self._embedding_cache_misses,
            ),
            "llm_hits": self._llm_cache_hits,
            "llm_misses": self._llm_cache_misses,
            "llm_hit_rate_pct": _rate(
                self._llm_cache_hits, self._llm_cache_misses,
            ),
        }
