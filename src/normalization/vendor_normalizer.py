"""Vendor name normalization and deduplication.

Real-world vendor names are incredibly messy.  This module resolves
raw name strings into canonical vendor entities using a three-stage
pipeline:

1. **Vendor-ID grouping** (fastest / most reliable when IDs are present).
2. **Cache lookup** (zero cost for previously-seen names).
3. **Fuzzy matching** against all known canonical names with auto-merge,
   pending-review, or new-entity decisions based on composite score
   thresholds.

All mappings are persisted in the SQLite cache so that normalization
quality improves with every run.
"""

import logging
import uuid
from collections import Counter
from typing import Optional

from rapidfuzz import fuzz

from src.normalization.cache import CacheManager
from src.utils.similarity import normalize_vendor_name

logger = logging.getLogger(__name__)

# Composite-score thresholds
_AUTO_MERGE_THRESHOLD: float = 90.0
_PENDING_REVIEW_THRESHOLD: float = 75.0

# Cross-reference duplicate-canonical detection
_CANONICAL_DEDUP_THRESHOLD: float = 85.0


def _generate_canonical_id() -> str:
    """Return a new canonical vendor ID."""
    return f"CV-{uuid.uuid4().hex[:10]}"


def _composite_score(name_a: str, name_b: str) -> float:
    """Weighted composite of token-sort-ratio and partial-ratio.

    Both inputs should already be normalised (lowered, suffix-stripped).
    Returns a score in [0, 100].
    """
    if not name_a or not name_b:
        return 0.0
    token_sort = fuzz.token_sort_ratio(name_a, name_b)
    partial = fuzz.partial_ratio(name_a, name_b)
    return 0.7 * token_sort + 0.3 * partial


class VendorNormalizer:
    """Normalize vendor names into canonical entities."""

    def __init__(self, cache: CacheManager) -> None:
        self.cache = cache

        # {canonical_id: canonical_name}
        self._canonical_vendors: dict[str, str] = {}
        # {normalised_name: canonical_id} — fast look-up index
        self._norm_index: dict[str, str] = {}

        self._load_cached_vendors()

        # Counters for reporting
        self._auto_merged: int = 0
        self._pending_reviews: list[dict] = []
        self._new_vendors: int = 0
        self._total_raw: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_cached_vendors(self) -> None:
        """Populate the in-memory canonical vendor set from the cache."""
        for cid, cname in self.cache.get_all_canonical_vendors():
            self._canonical_vendors[cid] = cname
            norm = normalize_vendor_name(cname)
            if norm:
                self._norm_index[norm] = cid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_all(
        self,
        vendor_names: list[str],
        vendor_ids: Optional[list[str]] = None,
    ) -> dict:
        """Normalize a list of vendor names into canonical entities.

        Returns a dict mapping each original name to::

            {canonical_id, canonical_name, confidence, is_new}
        """
        result: dict[str, dict] = {}
        self._total_raw += len(vendor_names)

        # ---- Step 1: vendor-ID grouping --------------------------------
        id_handled: set[str] = set()
        if vendor_ids is not None and len(vendor_ids) == len(vendor_names):
            result, id_handled = self._group_by_vendor_id(
                vendor_names, vendor_ids,
            )

        # ---- Step 2 + 3: cache / fuzzy for remaining names -------------
        for i, name in enumerate(vendor_names):
            if name in result:
                continue
            if name in id_handled:
                continue
            mapping = self._resolve_single(name)
            result[name] = mapping

        # ---- Step 4: cross-reference duplicate canonical detection ------
        self._cross_reference_check()

        return result

    def normalize_single(
        self, vendor_name: str, vendor_id: Optional[str] = None,
    ) -> dict:
        """Normalize a single vendor name (incremental processing)."""
        self._total_raw += 1

        # If we have a vendor_id, see if it's already canonical
        if vendor_id:
            cached = self.cache.get_canonical_vendor(vendor_id)
            if cached:
                return {
                    "canonical_id": cached[0],
                    "canonical_name": cached[1],
                    "confidence": cached[2],
                    "is_new": False,
                }

        return self._resolve_single(vendor_name)

    def get_canonical_id(self, vendor_name: str) -> Optional[str]:
        """Quick lookup — return canonical ID or ``None``."""
        cached = self.cache.get_canonical_vendor(vendor_name)
        if cached:
            return cached[0]
        norm = normalize_vendor_name(vendor_name)
        return self._norm_index.get(norm)

    def merge_vendors(self, vendor_id_keep: str, vendor_id_merge: str) -> None:
        """Merge *vendor_id_merge* into *vendor_id_keep*.

        All aliases that pointed to the merged vendor are re-pointed to
        the kept vendor.
        """
        if vendor_id_keep not in self._canonical_vendors:
            raise ValueError(f"Unknown canonical vendor: {vendor_id_keep}")
        if vendor_id_merge not in self._canonical_vendors:
            raise ValueError(f"Unknown canonical vendor: {vendor_id_merge}")

        keep_name = self._canonical_vendors[vendor_id_keep]
        merge_aliases = self.get_vendor_aliases(vendor_id_merge)

        for alias in merge_aliases:
            self.cache.set_vendor_alias(
                alias, vendor_id_keep, keep_name, 1.0, verified=True,
            )

        # Update in-memory state
        del self._canonical_vendors[vendor_id_merge]
        # Rebuild norm index
        self._norm_index = {
            normalize_vendor_name(name): cid
            for cid, name in self._canonical_vendors.items()
            if normalize_vendor_name(name)
        }
        logger.info(
            "Merged vendor %s into %s (%d aliases re-pointed)",
            vendor_id_merge, vendor_id_keep, len(merge_aliases),
        )

    def get_vendor_aliases(self, canonical_id: str) -> list[str]:
        """Return all known alias strings for a canonical vendor."""
        try:
            rows = self.cache._conn.execute(
                "SELECT alias_text FROM vendor_aliases "
                "WHERE canonical_vendor_id = ? ORDER BY alias_text",
                (canonical_id,),
            ).fetchall()
            return [r["alias_text"] for r in rows]
        except Exception:
            logger.exception("Failed to get aliases for %s", canonical_id)
            return []

    def get_normalization_report(self) -> dict:
        """Summary of the most recent normalization run."""
        return {
            "total_raw_names": self._total_raw,
            "total_canonical_vendors": len(self._canonical_vendors),
            "auto_merged_count": self._auto_merged,
            "pending_review_count": len(self._pending_reviews),
            "new_vendor_count": self._new_vendors,
            "pending_reviews": list(self._pending_reviews),
        }

    # ------------------------------------------------------------------
    # Internal: vendor-ID grouping
    # ------------------------------------------------------------------

    def _group_by_vendor_id(
        self,
        names: list[str],
        ids: list[str],
    ) -> tuple[dict, set[str]]:
        """Group names by vendor ID and pick the most common variant."""
        id_to_names: dict[str, list[str]] = {}
        for name, vid in zip(names, ids):
            if vid and str(vid).strip():
                id_to_names.setdefault(vid, []).append(name)

        result: dict[str, dict] = {}
        handled: set[str] = set()

        for vid, name_list in id_to_names.items():
            # Pick the most frequent name variant as canonical
            counter = Counter(name_list)
            canonical_name = counter.most_common(1)[0][0]
            canonical_id = f"ERP-{vid}"

            # Register in cache and memory
            self._canonical_vendors[canonical_id] = canonical_name
            norm = normalize_vendor_name(canonical_name)
            if norm:
                self._norm_index[norm] = canonical_id

            for name in set(name_list):
                self.cache.set_vendor_alias(
                    name, canonical_id, canonical_name, 1.0,
                )
                result[name] = {
                    "canonical_id": canonical_id,
                    "canonical_name": canonical_name,
                    "confidence": 1.0,
                    "is_new": False,
                }
                handled.add(name)

        return result, handled

    # ------------------------------------------------------------------
    # Internal: single-name resolution
    # ------------------------------------------------------------------

    def _resolve_single(self, vendor_name: str) -> dict:
        """Resolve one vendor name through cache → fuzzy → new-entity."""
        # ---- cache hit -------------------------------------------------
        cached = self.cache.get_canonical_vendor(vendor_name)
        if cached:
            return {
                "canonical_id": cached[0],
                "canonical_name": cached[1],
                "confidence": cached[2],
                "is_new": False,
            }

        # ---- fuzzy match -----------------------------------------------
        norm_input = normalize_vendor_name(vendor_name)
        if not norm_input:
            return self._create_new_entity(vendor_name)

        best_score: float = 0.0
        best_cid: Optional[str] = None
        best_cname: Optional[str] = None

        for cid, cname in self._canonical_vendors.items():
            norm_existing = normalize_vendor_name(cname)
            if not norm_existing:
                continue
            score = _composite_score(norm_input, norm_existing)
            if score > best_score:
                best_score = score
                best_cid = cid
                best_cname = cname

        # ---- decision --------------------------------------------------
        if best_score >= _AUTO_MERGE_THRESHOLD and best_cid and best_cname:
            confidence = 0.95
            self.cache.set_vendor_alias(
                vendor_name, best_cid, best_cname, confidence,
            )
            self._auto_merged += 1
            return {
                "canonical_id": best_cid,
                "canonical_name": best_cname,
                "confidence": confidence,
                "is_new": False,
            }

        if best_score >= _PENDING_REVIEW_THRESHOLD and best_cid and best_cname:
            confidence = 0.70
            self.cache.set_vendor_alias(
                vendor_name, best_cid, best_cname, confidence,
            )
            self._pending_reviews.append({
                "name_a": vendor_name,
                "name_b": best_cname,
                "score": round(best_score, 1),
            })
            self._auto_merged += 1
            return {
                "canonical_id": best_cid,
                "canonical_name": best_cname,
                "confidence": confidence,
                "is_new": False,
            }

        # ---- new entity ------------------------------------------------
        return self._create_new_entity(vendor_name)

    def _create_new_entity(self, vendor_name: str) -> dict:
        """Create a brand-new canonical vendor entity."""
        cid = _generate_canonical_id()
        self._canonical_vendors[cid] = vendor_name
        norm = normalize_vendor_name(vendor_name)
        if norm:
            self._norm_index[norm] = cid
        self.cache.set_vendor_alias(vendor_name, cid, vendor_name, 1.0)
        self._new_vendors += 1
        return {
            "canonical_id": cid,
            "canonical_name": vendor_name,
            "confidence": 1.0,
            "is_new": True,
        }

    # ------------------------------------------------------------------
    # Internal: cross-reference check
    # ------------------------------------------------------------------

    def _cross_reference_check(self) -> None:
        """Flag canonical entities whose names are suspiciously similar."""
        items = list(self._canonical_vendors.items())
        for i in range(len(items)):
            cid_a, name_a = items[i]
            norm_a = normalize_vendor_name(name_a)
            if not norm_a:
                continue
            for j in range(i + 1, len(items)):
                cid_b, name_b = items[j]
                norm_b = normalize_vendor_name(name_b)
                if not norm_b:
                    continue
                score = _composite_score(norm_a, norm_b)
                if score >= _CANONICAL_DEDUP_THRESHOLD:
                    # Avoid logging duplicates already captured
                    already = any(
                        (p["name_a"] == name_a and p["name_b"] == name_b)
                        or (p["name_a"] == name_b and p["name_b"] == name_a)
                        for p in self._pending_reviews
                    )
                    if not already:
                        self._pending_reviews.append({
                            "name_a": name_a,
                            "name_b": name_b,
                            "score": round(score, 1),
                        })
                        logger.info(
                            "Potential duplicate canonicals: %r ↔ %r (%.1f)",
                            name_a, name_b, score,
                        )
