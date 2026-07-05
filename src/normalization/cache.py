"""SQLite-backed cache system for SilentAuditor.

Central persistence layer for vendor normalizations, item classifications,
embeddings, LLM responses, feedback, and cost tracking. Every function that
could call the LLM must check cache first.
"""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _hash_text(text: str) -> str:
    """Return SHA-256 hex digest of the given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CacheManager:
    """SQLite-backed cache used across all SilentAuditor modules.

    Stores vendor aliases, item classifications, embeddings, LLM responses,
    user feedback, field mappings, vendor baselines, and LLM cost tracking.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = str(settings.DATABASE_PATH)

        self._db_path = db_path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._create_tables()
        logger.info("CacheManager initialised with database at %s", self._db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Create all cache tables if they do not exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS vendor_aliases (
                alias_text        TEXT PRIMARY KEY,
                canonical_vendor_id TEXT NOT NULL,
                canonical_name    TEXT NOT NULL,
                confidence        REAL NOT NULL,
                verified_by_human BOOLEAN DEFAULT FALSE,
                created_at        TIMESTAMP,
                updated_at        TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS item_classifications (
                description_hash       TEXT PRIMARY KEY,
                original_description   TEXT,
                normalized_description TEXT,
                category               TEXT,
                subcategory            TEXT,
                item_type              TEXT,
                unit_of_measure        TEXT,
                classification_source  TEXT,
                confidence             REAL,
                created_at             TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash     TEXT PRIMARY KEY,
                original_text TEXT,
                embedding     BLOB,
                model_name    TEXT,
                created_at    TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS llm_cache (
                prompt_hash   TEXT PRIMARY KEY,
                prompt_text   TEXT,
                response_json TEXT,
                model         TEXT,
                input_tokens  INTEGER,
                output_tokens INTEGER,
                cost_usd      REAL,
                created_at    TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS feedback (
                finding_id   TEXT PRIMARY KEY,
                customer_id  TEXT,
                verdict      TEXT,
                notes        TEXT,
                created_at   TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS field_mappings (
                customer_id   TEXT,
                source_column TEXT,
                target_field  TEXT,
                confidence    REAL,
                verified      BOOLEAN,
                PRIMARY KEY (customer_id, source_column)
            );

            CREATE TABLE IF NOT EXISTS vendor_baselines (
                vendor_id     TEXT,
                customer_id   TEXT,
                baseline_data TEXT,
                calculated_at TIMESTAMP,
                PRIMARY KEY (vendor_id, customer_id)
            );

            CREATE TABLE IF NOT EXISTS llm_cost_tracking (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id   TEXT,
                module        TEXT,
                input_tokens  INTEGER,
                output_tokens INTEGER,
                cost_usd      REAL,
                created_at    TIMESTAMP
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Vendor alias operations
    # ------------------------------------------------------------------

    def get_canonical_vendor(
        self, alias_text: str
    ) -> Optional[tuple[str, str, float]]:
        """Look up canonical vendor for an alias.

        Returns (canonical_vendor_id, canonical_name, confidence) or None.
        """
        try:
            row = self._conn.execute(
                "SELECT canonical_vendor_id, canonical_name, confidence "
                "FROM vendor_aliases WHERE alias_text = ?",
                (alias_text,),
            ).fetchone()
            if row is None:
                return None
            return (row["canonical_vendor_id"], row["canonical_name"], row["confidence"])
        except sqlite3.Error:
            logger.exception("Failed to look up vendor alias %r", alias_text)
            return None

    def set_vendor_alias(
        self,
        alias_text: str,
        canonical_id: str,
        canonical_name: str,
        confidence: float,
        verified: bool = False,
    ) -> None:
        """Insert or update a vendor alias mapping."""
        now = _utcnow()
        try:
            self._conn.execute(
                "INSERT INTO vendor_aliases "
                "(alias_text, canonical_vendor_id, canonical_name, confidence, "
                " verified_by_human, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(alias_text) DO UPDATE SET "
                " canonical_vendor_id = excluded.canonical_vendor_id, "
                " canonical_name = excluded.canonical_name, "
                " confidence = excluded.confidence, "
                " verified_by_human = excluded.verified_by_human, "
                " updated_at = excluded.updated_at",
                (alias_text, canonical_id, canonical_name, confidence, verified, now, now),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to set vendor alias %r", alias_text)

    def get_all_canonical_vendors(self) -> list[tuple[str, str]]:
        """Return all distinct (canonical_vendor_id, canonical_name) pairs."""
        try:
            rows = self._conn.execute(
                "SELECT DISTINCT canonical_vendor_id, canonical_name "
                "FROM vendor_aliases ORDER BY canonical_name"
            ).fetchall()
            return [(r["canonical_vendor_id"], r["canonical_name"]) for r in rows]
        except sqlite3.Error:
            logger.exception("Failed to fetch canonical vendors")
            return []

    def verify_vendor_alias(self, alias_text: str) -> None:
        """Mark a vendor alias as human-verified."""
        try:
            self._conn.execute(
                "UPDATE vendor_aliases SET verified_by_human = TRUE, updated_at = ? "
                "WHERE alias_text = ?",
                (_utcnow(), alias_text),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to verify vendor alias %r", alias_text)

    # ------------------------------------------------------------------
    # Item classification operations
    # ------------------------------------------------------------------

    def get_item_classification(self, description: str) -> Optional[dict]:
        """Look up a cached item classification by description.

        Returns a dict with classification fields or None.
        """
        desc_hash = _hash_text(description)
        try:
            row = self._conn.execute(
                "SELECT normalized_description, category, subcategory, item_type, "
                "unit_of_measure, classification_source, confidence "
                "FROM item_classifications WHERE description_hash = ?",
                (desc_hash,),
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        except sqlite3.Error:
            logger.exception("Failed to look up item classification")
            return None

    def set_item_classification(
        self,
        description: str,
        normalized: str,
        category: str,
        subcategory: Optional[str],
        item_type: Optional[str],
        unit: Optional[str],
        source: str,
        confidence: float,
    ) -> None:
        """Cache an item classification result."""
        desc_hash = _hash_text(description)
        try:
            self._conn.execute(
                "INSERT INTO item_classifications "
                "(description_hash, original_description, normalized_description, "
                " category, subcategory, item_type, unit_of_measure, "
                " classification_source, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(description_hash) DO UPDATE SET "
                " normalized_description = excluded.normalized_description, "
                " category = excluded.category, "
                " subcategory = excluded.subcategory, "
                " item_type = excluded.item_type, "
                " unit_of_measure = excluded.unit_of_measure, "
                " classification_source = excluded.classification_source, "
                " confidence = excluded.confidence",
                (desc_hash, description, normalized, category, subcategory,
                 item_type, unit, source, confidence, _utcnow()),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to set item classification for %r", description)

    # ------------------------------------------------------------------
    # Embedding operations
    # ------------------------------------------------------------------

    def get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Retrieve a cached embedding vector.

        Returns a numpy array or None if not cached.
        """
        text_hash = _hash_text(f"{model_name}:{text}")
        try:
            row = self._conn.execute(
                "SELECT embedding FROM embedding_cache "
                "WHERE text_hash = ? AND model_name = ?",
                (text_hash, model_name),
            ).fetchone()
            if row is None:
                return None
            return np.frombuffer(row["embedding"], dtype=np.float32).copy()
        except sqlite3.Error:
            logger.exception("Failed to look up embedding")
            return None

    def set_embedding(self, text: str, embedding: np.ndarray, model_name: str) -> None:
        """Cache an embedding vector."""
        text_hash = _hash_text(f"{model_name}:{text}")
        blob = embedding.astype(np.float32).tobytes()
        try:
            self._conn.execute(
                "INSERT INTO embedding_cache "
                "(text_hash, original_text, embedding, model_name, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(text_hash) DO UPDATE SET "
                " embedding = excluded.embedding, "
                " model_name = excluded.model_name",
                (text_hash, text, blob, model_name, _utcnow()),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to cache embedding")

    # ------------------------------------------------------------------
    # LLM cache operations
    # ------------------------------------------------------------------

    def get_llm_response(self, prompt: str) -> Optional[dict]:
        """Retrieve a cached LLM response.

        Returns parsed JSON response or None.
        """
        prompt_hash = _hash_text(prompt)
        try:
            row = self._conn.execute(
                "SELECT response_json FROM llm_cache WHERE prompt_hash = ?",
                (prompt_hash,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row["response_json"])
        except sqlite3.Error:
            logger.exception("Failed to look up LLM cache")
            return None

    def set_llm_response(
        self,
        prompt: str,
        response_json: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """Cache an LLM response."""
        prompt_hash = _hash_text(prompt)
        try:
            self._conn.execute(
                "INSERT INTO llm_cache "
                "(prompt_hash, prompt_text, response_json, model, "
                " input_tokens, output_tokens, cost_usd, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(prompt_hash) DO UPDATE SET "
                " response_json = excluded.response_json, "
                " model = excluded.model, "
                " input_tokens = excluded.input_tokens, "
                " output_tokens = excluded.output_tokens, "
                " cost_usd = excluded.cost_usd",
                (prompt_hash, prompt, response_json, model,
                 input_tokens, output_tokens, cost_usd, _utcnow()),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to cache LLM response")

    # ------------------------------------------------------------------
    # Feedback operations
    # ------------------------------------------------------------------

    def record_feedback(
        self,
        finding_id: str,
        customer_id: str,
        verdict: str,
        notes: Optional[str],
    ) -> None:
        """Record user feedback on a finding."""
        try:
            self._conn.execute(
                "INSERT INTO feedback "
                "(finding_id, customer_id, verdict, notes, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(finding_id) DO UPDATE SET "
                " verdict = excluded.verdict, "
                " notes = excluded.notes",
                (finding_id, customer_id, verdict, notes, _utcnow()),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to record feedback for %s", finding_id)

    def get_feedback_for_pattern(
        self, module: str, vendor_id: str
    ) -> list[dict]:
        """Return all feedback entries matching a module/vendor pattern.

        Searches feedback joined with finding metadata pattern. Since feedback
        stores finding_id only, we match by customer feedback entries whose
        finding_id contains the module name or vendor_id.
        """
        try:
            rows = self._conn.execute(
                "SELECT finding_id, customer_id, verdict, notes, created_at "
                "FROM feedback "
                "WHERE finding_id LIKE ? OR finding_id LIKE ? "
                "ORDER BY created_at DESC",
                (f"%{module}%", f"%{vendor_id}%"),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            logger.exception("Failed to get feedback for pattern")
            return []

    def get_false_positive_count(
        self,
        module: str,
        pattern_hash: str,
        customer_id: Optional[str] = None,
    ) -> int:
        """Count false-positive verdicts for a given module/pattern."""
        try:
            if customer_id:
                row = self._conn.execute(
                    "SELECT COUNT(*) as cnt FROM feedback "
                    "WHERE verdict = 'false_positive' "
                    "AND customer_id = ? "
                    "AND (finding_id LIKE ? OR finding_id LIKE ?)",
                    (customer_id, f"%{module}%", f"%{pattern_hash}%"),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) as cnt FROM feedback "
                    "WHERE verdict = 'false_positive' "
                    "AND (finding_id LIKE ? OR finding_id LIKE ?)",
                    (f"%{module}%", f"%{pattern_hash}%"),
                ).fetchone()
            return row["cnt"] if row else 0
        except sqlite3.Error:
            logger.exception("Failed to count false positives")
            return 0

    # ------------------------------------------------------------------
    # Field mapping operations
    # ------------------------------------------------------------------

    def get_field_mappings(self, customer_id: str) -> dict:
        """Return all field mappings for a customer as {source_column: target_field}."""
        try:
            rows = self._conn.execute(
                "SELECT source_column, target_field, confidence, verified "
                "FROM field_mappings WHERE customer_id = ?",
                (customer_id,),
            ).fetchall()
            return {
                r["source_column"]: {
                    "target_field": r["target_field"],
                    "confidence": r["confidence"],
                    "verified": bool(r["verified"]),
                }
                for r in rows
            }
        except sqlite3.Error:
            logger.exception("Failed to get field mappings for %s", customer_id)
            return {}

    def set_field_mappings(self, customer_id: str, mappings: dict) -> None:
        """Store field mappings for a customer.

        mappings: {source_column: {"target_field": str, "confidence": float, "verified": bool}}
        """
        try:
            for source_col, mapping in mappings.items():
                self._conn.execute(
                    "INSERT INTO field_mappings "
                    "(customer_id, source_column, target_field, confidence, verified) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(customer_id, source_column) DO UPDATE SET "
                    " target_field = excluded.target_field, "
                    " confidence = excluded.confidence, "
                    " verified = excluded.verified",
                    (
                        customer_id,
                        source_col,
                        mapping["target_field"],
                        mapping["confidence"],
                        mapping.get("verified", False),
                    ),
                )
            self._conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to set field mappings for %s", customer_id)

    # ------------------------------------------------------------------
    # Vendor baseline operations
    # ------------------------------------------------------------------

    def get_vendor_baseline(
        self, vendor_id: str, customer_id: str
    ) -> Optional[dict]:
        """Retrieve stored vendor baseline data."""
        try:
            row = self._conn.execute(
                "SELECT baseline_data, calculated_at FROM vendor_baselines "
                "WHERE vendor_id = ? AND customer_id = ?",
                (vendor_id, customer_id),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row["baseline_data"])
        except sqlite3.Error:
            logger.exception("Failed to get vendor baseline")
            return None

    def set_vendor_baseline(
        self, vendor_id: str, customer_id: str, baseline_data: dict
    ) -> None:
        """Store or update vendor baseline data."""
        try:
            self._conn.execute(
                "INSERT INTO vendor_baselines "
                "(vendor_id, customer_id, baseline_data, calculated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(vendor_id, customer_id) DO UPDATE SET "
                " baseline_data = excluded.baseline_data, "
                " calculated_at = excluded.calculated_at",
                (vendor_id, customer_id, json.dumps(baseline_data), _utcnow()),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to set vendor baseline")

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------

    def log_llm_cost(
        self,
        customer_id: str,
        module: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """Log an LLM API call cost."""
        try:
            self._conn.execute(
                "INSERT INTO llm_cost_tracking "
                "(customer_id, module, input_tokens, output_tokens, cost_usd, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (customer_id, module, input_tokens, output_tokens, cost_usd, _utcnow()),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to log LLM cost")

    def get_monthly_cost(self, customer_id: str) -> Decimal:
        """Return total LLM cost for the current calendar month."""
        try:
            now = datetime.now(timezone.utc)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) as total "
                "FROM llm_cost_tracking "
                "WHERE customer_id = ? AND created_at >= ?",
                (customer_id, month_start),
            ).fetchone()
            return Decimal(str(row["total"]))
        except sqlite3.Error:
            logger.exception("Failed to get monthly cost for %s", customer_id)
            return Decimal("0")

    def get_cost_by_module(self, customer_id: str) -> dict:
        """Return LLM costs broken down by module for the current month."""
        try:
            now = datetime.now(timezone.utc)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
            rows = self._conn.execute(
                "SELECT module, SUM(cost_usd) as total, "
                "SUM(input_tokens) as total_input, SUM(output_tokens) as total_output "
                "FROM llm_cost_tracking "
                "WHERE customer_id = ? AND created_at >= ? "
                "GROUP BY module",
                (customer_id, month_start),
            ).fetchall()
            return {
                r["module"]: {
                    "cost_usd": Decimal(str(r["total"])),
                    "input_tokens": r["total_input"],
                    "output_tokens": r["total_output"],
                }
                for r in rows
            }
        except sqlite3.Error:
            logger.exception("Failed to get cost by module for %s", customer_id)
            return {}

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def vacuum(self) -> None:
        """Compact the database file."""
        try:
            self._conn.execute("VACUUM")
            logger.info("Database vacuumed successfully")
        except sqlite3.Error:
            logger.exception("Failed to vacuum database")

    def get_cache_stats(self) -> dict:
        """Return row counts and database size."""
        tables = [
            "vendor_aliases",
            "item_classifications",
            "embedding_cache",
            "llm_cache",
            "feedback",
            "field_mappings",
            "vendor_baselines",
            "llm_cost_tracking",
        ]
        stats: dict = {"tables": {}}
        try:
            for table in tables:
                row = self._conn.execute(
                    f"SELECT COUNT(*) as cnt FROM {table}"  # noqa: S608 — table names are hardcoded
                ).fetchone()
                stats["tables"][table] = row["cnt"]

            db_path = Path(self._db_path)
            stats["database_size_bytes"] = db_path.stat().st_size if db_path.exists() else 0
            stats["total_rows"] = sum(stats["tables"].values())
        except sqlite3.Error:
            logger.exception("Failed to get cache stats")
        return stats

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
            logger.info("CacheManager connection closed")
        except sqlite3.Error:
            logger.exception("Failed to close database connection")
