"""Abstract base class for all anomaly detection modules.

Every detection module inherits from :class:`BaseDetector`, which
enforces a consistent interface and provides shared helpers for
creating findings, checking data readiness, and tracking statistics.

:class:`ModuleRunner` orchestrates execution of all registered modules
against a dataset, collecting findings and per-module stats.
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import pandas as pd

from src.normalization.cache import CacheManager
from src.utils.constants import (
    Finding,
    ModuleName,
    Severity,
    generate_finding_id,
)

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Abstract base class for all anomaly detection modules."""

    def __init__(
        self,
        config: dict,
        cache: CacheManager,
        llm_client: object,
    ) -> None:
        self.config = config
        self.cache = cache
        self.llm_client = llm_client
        self.findings: list[Finding] = []
        self.stats: dict = {
            "invoices_analyzed": 0,
            "findings_generated": 0,
            "tier1_calls": 0,
            "tier2_calls": 0,
            "tier3_calls": 0,
            "processing_time_seconds": 0.0,
        }
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def detect(
        self,
        invoice_data: pd.DataFrame,
        vendor_data: Optional[pd.DataFrame] = None,
        supplementary_data: Optional[dict] = None,
    ) -> list[Finding]:
        """Run detection on the provided data.

        Args:
            invoice_data: Normalised invoice DataFrame.
            vendor_data: Optional vendor master DataFrame.
            supplementary_data: Optional dict with keys such as
                ``'contracts'``, ``'receipts'``,
                ``'approval_thresholds'``.

        Returns:
            List of :class:`Finding` objects.
        """

    @abstractmethod
    def get_module_name(self) -> ModuleName:
        """Return this module's :class:`ModuleName` enum value."""

    @abstractmethod
    def get_required_fields(self) -> list[str]:
        """Return DataFrame column names required for this module."""

    @abstractmethod
    def get_optional_fields(self) -> list[str]:
        """Return DataFrame column names that enhance detection."""

    # ------------------------------------------------------------------
    # Data-readiness check
    # ------------------------------------------------------------------

    def can_run(self, available_fields: list[str]) -> tuple[bool, float]:
        """Check whether the module can run on the available data.

        Returns:
            ``(can_run, effectiveness)`` where *effectiveness* is a float
            in [0.0, 1.0] indicating how well the module will perform.
        """
        required = set(self.get_required_fields())
        optional = set(self.get_optional_fields())
        available = set(available_fields)

        missing_required = required - available
        if missing_required:
            self.logger.warning(
                "Cannot run %s: missing required fields %s",
                self.get_module_name().value, missing_required,
            )
            return False, 0.0

        if optional:
            optional_present = optional & available
            effectiveness = 0.6 + 0.4 * (len(optional_present) / len(optional))
        else:
            effectiveness = 1.0

        return True, round(effectiveness, 2)

    # ------------------------------------------------------------------
    # Finding creation
    # ------------------------------------------------------------------

    def create_finding(
        self,
        severity: Severity,
        confidence: float,
        vendor_id: str,
        vendor_name: str,
        invoice_ids: list[str],
        amount_at_risk: Decimal,
        description: str,
        evidence: dict,
        recommended_action: str,
    ) -> Finding:
        """Create a :class:`Finding` and append it to ``self.findings``."""
        finding = Finding(
            finding_id=generate_finding_id(),
            module=self.get_module_name(),
            severity=severity,
            confidence=confidence,
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            invoice_ids=invoice_ids,
            amount_at_risk=amount_at_risk,
            description=description,
            evidence=evidence,
            recommended_action=recommended_action,
            module_correlations=[],
            created_at=datetime.now(timezone.utc),
            suppressed=False,
            suppression_reason=None,
        )
        self.findings.append(finding)
        self.stats["findings_generated"] += 1
        return finding

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return processing statistics for this module."""
        return {**self.stats, "module": self.get_module_name().value}


# ======================================================================
# Module runner
# ======================================================================

class ModuleRunner:
    """Orchestrate running multiple detection modules."""

    def __init__(
        self,
        config: dict,
        cache: CacheManager,
        llm_client: object,
    ) -> None:
        self.config = config
        self.cache = cache
        self.llm_client = llm_client
        self.modules: list[BaseDetector] = []

    def register_module(self, module_class: type) -> None:
        """Instantiate and register a detection module class.

        *module_class* must be a concrete subclass of
        :class:`BaseDetector`.  The appropriate section of
        ``self.config`` is passed to the module based on its
        :meth:`~BaseDetector.get_module_name`.
        """
        # Instantiate with the full config first to read the module name,
        # then narrow config to the module-specific section if available.
        instance = module_class(self.config, self.cache, self.llm_client)
        module_key = instance.get_module_name().value
        module_config = self.config.get(module_key, self.config)
        instance.config = module_config
        self.modules.append(instance)
        logger.info("Registered module: %s", module_key)

    def run_all(
        self,
        invoice_data: pd.DataFrame,
        vendor_data: Optional[pd.DataFrame] = None,
        supplementary_data: Optional[dict] = None,
    ) -> list[Finding]:
        """Run every registered module that can operate on the data.

        Modules whose required fields are missing are skipped with a
        log message.  Returns the combined findings list.
        """
        all_findings: list[Finding] = []
        available_fields = list(invoice_data.columns)

        for module in self.modules:
            name = module.get_module_name().value
            can_run, effectiveness = module.can_run(available_fields)

            if not can_run:
                logger.info(
                    "Skipping %s — missing required fields", name,
                )
                continue

            logger.info(
                "Running %s (effectiveness %.0f%%)...",
                name, effectiveness * 100,
            )
            start = time.perf_counter()
            try:
                findings = module.detect(
                    invoice_data, vendor_data, supplementary_data,
                )
                elapsed = time.perf_counter() - start
                module.stats["processing_time_seconds"] = round(elapsed, 3)
                all_findings.extend(findings)
                logger.info(
                    "  %s completed: %d findings in %.2fs",
                    name, len(findings), elapsed,
                )
            except Exception:
                elapsed = time.perf_counter() - start
                module.stats["processing_time_seconds"] = round(elapsed, 3)
                logger.exception("  %s failed after %.2fs", name, elapsed)

        # --- Cross-module deduplication ---
        # Group by (invoice_id, vendor_id). For groups with 2+ findings,
        # keep highest-confidence, annotate with corroborating_modules.
        all_findings = self._cross_module_dedup(all_findings)

        # --- Minimum confidence gate ---
        pre_gate = len(all_findings)
        all_findings = [f for f in all_findings if f.confidence >= 0.50]
        gated = pre_gate - len(all_findings)
        if gated > 0:
            logger.info(
                "Confidence gate removed %d findings below 0.50", gated,
            )

        logger.info(
            "All modules done. Total findings: %d", len(all_findings),
        )
        return all_findings

    @staticmethod
    def _cross_module_dedup(findings: list[Finding]) -> list[Finding]:
        """Deduplicate findings across modules.

        For each (invoice_id, vendor_id) pair with findings from multiple
        modules, keep the highest-confidence finding and annotate it with
        a list of corroborating modules.
        """
        from collections import defaultdict

        # Group by (first_invoice_id, vendor_id)
        groups: dict[tuple[str, str], list[Finding]] = defaultdict(list)
        ungrouped: list[Finding] = []

        for f in findings:
            if f.suppressed:
                ungrouped.append(f)
                continue
            inv_id = f.invoice_ids[0] if f.invoice_ids else ""
            vid = f.vendor_id or ""
            if inv_id and vid:
                groups[(inv_id, vid)].append(f)
            else:
                ungrouped.append(f)

        deduped: list[Finding] = list(ungrouped)
        for key, group in groups.items():
            if len(group) <= 1:
                deduped.extend(group)
                continue

            # Multiple modules flagged the same (invoice, vendor)
            modules_present = list({f.module.value for f in group})
            if len(modules_present) <= 1:
                # Same module — already handled by intra-module dedup
                deduped.extend(group)
                continue

            # Keep highest-confidence finding, annotate with corroborating modules
            group.sort(key=lambda f: f.confidence, reverse=True)
            survivor = group[0]
            survivor.corroborating_modules = modules_present
            survivor.corroboration_count = len(modules_present)
            deduped.append(survivor)

        return deduped

    def get_all_stats(self) -> dict:
        """Return per-module and aggregate statistics."""
        module_stats = [m.get_stats() for m in self.modules]
        total_findings = sum(s["findings_generated"] for s in module_stats)
        total_time = sum(s["processing_time_seconds"] for s in module_stats)
        return {
            "modules": module_stats,
            "total_findings": total_findings,
            "total_processing_time_seconds": round(total_time, 3),
        }
