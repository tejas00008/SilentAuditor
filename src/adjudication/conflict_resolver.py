"""Resolve contradictions and amplify corroboration between module findings.

Six resolution rules:

1. Duplicate vs Contract — downgrade if contract shows separate milestones.
2. Price Creep vs Market — suppress if market trend explains the increase.
3. Split vs Vendor Behavior — suppress if vendor's baseline is consistent.
4. Phantom vs Contract — suppress if amended scope covers the service.
5. Multi-module corroboration — escalate when 2+ modules flag the same vendor.
6. Collusion + pricing — escalate all related findings to CRITICAL.
"""

import logging
from collections import defaultdict
from typing import Optional

from src.utils.constants import Finding, ModuleName, Severity

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = [Severity.INFORMATIONAL, Severity.REVIEW, Severity.CRITICAL]


def _escalate(current: Severity) -> Severity:
    """Return the next severity tier up, capped at CRITICAL."""
    idx = _SEVERITY_ORDER.index(current)
    return _SEVERITY_ORDER[min(idx + 1, len(_SEVERITY_ORDER) - 1)]


class ConflictResolver:
    """Resolve contradictions and amplify corroboration across findings."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self.resolution_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, findings: list[Finding]) -> list[Finding]:
        """Process all findings and return the updated list."""
        self.resolution_log = []

        # Index structures
        by_vendor: dict[str, list[Finding]] = defaultdict(list)
        by_invoice: dict[str, list[Finding]] = defaultdict(list)
        for f in findings:
            by_vendor[f.vendor_id].append(f)
            for inv_id in f.invoice_ids:
                by_invoice[inv_id].append(f)

        # Apply pairwise conflict rules first (may suppress)
        self._rule1_duplicate_vs_contract(by_invoice)
        self._rule2_price_creep_vs_market(by_vendor)
        self._rule3_split_vs_behavior(by_vendor)
        self._rule4_phantom_vs_contract(by_vendor)

        # Amplification rules (run on non-suppressed findings)
        self._rule5_multi_module_corroboration(by_vendor)
        self._rule6_collusion_plus_pricing(by_vendor, findings)

        return findings

    def get_resolution_report(self) -> dict:
        """Summary of all resolutions applied."""
        suppressions = [r for r in self.resolution_log if r["action"] == "suppress"]
        downgrades = [r for r in self.resolution_log if r["action"] == "downgrade"]
        escalations = [r for r in self.resolution_log if r["action"] == "escalate"]
        correlations = [r for r in self.resolution_log if r["action"] == "correlate"]

        return {
            "total_resolutions": len(self.resolution_log),
            "suppressions": len(suppressions),
            "downgrades": len(downgrades),
            "escalations": len(escalations),
            "correlations": len(correlations),
            "details": list(self.resolution_log),
        }

    # ------------------------------------------------------------------
    # Rule helpers
    # ------------------------------------------------------------------

    def _log(self, action: str, rule: str, finding_ids: list[str],
             reason: str) -> None:
        self.resolution_log.append({
            "action": action,
            "rule": rule,
            "finding_ids": finding_ids,
            "reason": reason,
        })

    @staticmethod
    def _modules_for_vendor(
        by_vendor: dict[str, list[Finding]],
        vendor_id: str,
        module: ModuleName,
        suppressed_ok: bool = False,
    ) -> list[Finding]:
        return [
            f for f in by_vendor.get(vendor_id, [])
            if f.module == module and (suppressed_ok or not f.suppressed)
        ]

    # ==================================================================
    # Rule 1 — Duplicate vs Contract Compliance
    # ==================================================================

    def _rule1_duplicate_vs_contract(
        self, by_invoice: dict[str, list[Finding]],
    ) -> None:
        for inv_id, findings in by_invoice.items():
            dups = [f for f in findings
                    if f.module == ModuleName.DUPLICATE_DETECTION and not f.suppressed]
            contracts = [f for f in findings
                         if f.module == ModuleName.CONTRACT_COMPLIANCE and not f.suppressed]
            if not dups or not contracts:
                continue

            # If contract findings suggest separate deliverables, downgrade
            for dup in dups:
                contract_descs = [c.description for c in contracts]
                if any("scope" in d.lower() or "rate" in d.lower()
                       or "milestone" in d.lower() for d in contract_descs):
                    continue  # contract flags problems too → keep duplicate
                # Contract exists and doesn't flag this invoice →
                # invoices may be separate milestones
                dup.confidence = round(dup.confidence * 0.6, 2)
                dup.description += (
                    " [Adj: Contract data suggests these may be "
                    "separate milestones]"
                )
                self._log(
                    "downgrade", "rule1_dup_vs_contract",
                    [dup.finding_id], "Contract suggests separate milestones",
                )

    # ==================================================================
    # Rule 2 — Price Creep vs Market Price
    # ==================================================================

    def _rule2_price_creep_vs_market(
        self, by_vendor: dict[str, list[Finding]],
    ) -> None:
        for vid, findings in by_vendor.items():
            creep = [f for f in findings
                     if f.module == ModuleName.PRICE_CREEP and not f.suppressed]
            market = [f for f in findings
                      if f.module == ModuleName.MARKET_PRICE and not f.suppressed]
            if not creep or not market:
                continue

            # If market price shows this vendor is NOT above market, the
            # price creep is explained by market-wide movement.
            market_methods = {f.evidence.get("method") for f in market}
            # If vendor doesn't appear in deviation findings → market is fine
            deviation_findings = [
                f for f in market
                if f.evidence.get("method") == "deviation_scoring"
            ]
            if deviation_findings:
                continue  # vendor IS above market → keep price creep

            for cf in creep:
                cf.suppressed = True
                cf.suppression_reason = "Vendor increase below market trend"
                self._log(
                    "suppress", "rule2_creep_vs_market",
                    [cf.finding_id], "Market trend explains price increase",
                )

    # ==================================================================
    # Rule 3 — Split Invoicing vs Vendor Behavior
    # ==================================================================

    def _rule3_split_vs_behavior(
        self, by_vendor: dict[str, list[Finding]],
    ) -> None:
        for vid, findings in by_vendor.items():
            splits = [f for f in findings
                      if f.module == ModuleName.SPLIT_INVOICING and not f.suppressed]
            behaviors = [f for f in findings
                         if f.module == ModuleName.VENDOR_BEHAVIOR and not f.suppressed]
            if not splits or not behaviors:
                continue

            # If vendor behavior shows NO anomaly (i.e. the behavior
            # findings are informational drift or normal-range) then the
            # split pattern is the vendor's baseline.
            # Heuristic: if all behavior findings are INFORMATIONAL, the
            # vendor is behaving normally.
            severe_behavior = [
                f for f in behaviors
                if f.severity in (Severity.CRITICAL, Severity.REVIEW)
            ]
            if severe_behavior:
                continue  # behaviour is itself anomalous → keep splits

            for sf in splits:
                sf.suppressed = True
                sf.suppression_reason = (
                    "Vendor's baseline pattern is consistent with flagged behavior"
                )
                self._log(
                    "suppress", "rule3_split_vs_behavior",
                    [sf.finding_id],
                    "Vendor's normal invoicing pattern matches split pattern",
                )

    # ==================================================================
    # Rule 4 — Phantom Services vs Contract Compliance
    # ==================================================================

    def _rule4_phantom_vs_contract(
        self, by_vendor: dict[str, list[Finding]],
    ) -> None:
        for vid, findings in by_vendor.items():
            phantoms = [f for f in findings
                        if f.module == ModuleName.PHANTOM_SERVICES and not f.suppressed]
            contracts = [f for f in findings
                         if f.module == ModuleName.CONTRACT_COMPLIANCE and not f.suppressed]
            if not phantoms or not contracts:
                continue

            # If contract compliance does NOT flag scope issues, the
            # contract covers this service.
            scope_flags = [
                f for f in contracts
                if f.evidence.get("check") == "scope_boundary"
            ]
            if scope_flags:
                continue  # contract itself says out-of-scope → keep phantom

            for pf in phantoms:
                if pf.evidence.get("method") in ("category_mismatch", "one_off_charge"):
                    pf.suppressed = True
                    pf.suppression_reason = "Service within amended contract scope"
                    self._log(
                        "suppress", "rule4_phantom_vs_contract",
                        [pf.finding_id],
                        "Contract scope covers this service category",
                    )

    # ==================================================================
    # Rule 5 — Multi-Module Corroboration
    # ==================================================================

    def _rule5_multi_module_corroboration(
        self, by_vendor: dict[str, list[Finding]],
    ) -> None:
        min_corroboration = 3  # Raised from 2 to 3 distinct modules
        min_confidence = 0.75  # Quality gate: only count module if best finding >= 0.75

        for vid, findings in by_vendor.items():
            active = [f for f in findings if not f.suppressed]
            if len(active) < 2:
                continue

            # Group by module
            modules_present: dict[ModuleName, list[Finding]] = defaultdict(list)
            for f in active:
                modules_present[f.module].append(f)

            # Quality gate: only count a module toward corroboration if its
            # best finding for this vendor has confidence >= min_confidence
            qualified_modules: set[ModuleName] = set()
            for mod, mod_findings in modules_present.items():
                best_conf = max(f.confidence for f in mod_findings)
                if best_conf >= min_confidence:
                    qualified_modules.add(mod)

            n_qualified = len(qualified_modules)
            if n_qualified < min_corroboration:
                continue

            module_names = sorted(m.value for m in qualified_modules)
            ids_linked = [f.finding_id for f in active]

            # Link findings — set corroborated flag but do NOT escalate severity
            for f in active:
                other_ids = [fid for fid in ids_linked if fid != f.finding_id]
                f.module_correlations = list(set(
                    f.module_correlations + other_ids
                ))

                # Instead of escalating severity, just annotate with corroboration info
                f.description += (
                    f" [Corroborated by {n_qualified} modules: "
                    + ", ".join(module_names) + "]"
                )

            self._log(
                "correlate", "rule5_multi_module",
                ids_linked,
                f"Vendor {vid} flagged by {n_qualified} qualified modules: "
                + ", ".join(module_names),
            )

    # ==================================================================
    # Rule 6 — Collusion + Pricing
    # ==================================================================

    def _rule6_collusion_plus_pricing(
        self,
        by_vendor: dict[str, list[Finding]],
        all_findings: list[Finding],
    ) -> None:
        # Find collusion cluster vendor IDs
        collusion_findings = [
            f for f in all_findings
            if f.module == ModuleName.VENDOR_COLLUSION and not f.suppressed
        ]
        if not collusion_findings:
            return

        # Collect all vendor IDs in collusion clusters
        cluster_vids: set[str] = set()
        for cf in collusion_findings:
            cluster_vids.add(cf.vendor_id)
            va = cf.evidence.get("vendor_a")
            vb = cf.evidence.get("vendor_b")
            if va:
                cluster_vids.add(va)
            if vb:
                cluster_vids.add(vb)
            for v in cf.evidence.get("cluster", []):
                cluster_vids.add(v)

        # Check if any cluster vendor has pricing findings
        pricing_modules = {ModuleName.PRICE_CREEP, ModuleName.MARKET_PRICE}
        escalated_ids: list[str] = []

        for vid in cluster_vids:
            vendor_findings = by_vendor.get(vid, [])
            pricing = [
                f for f in vendor_findings
                if f.module in pricing_modules and not f.suppressed
            ]
            if not pricing:
                continue

            # Escalate ALL findings for all vendors in this cluster
            for cv in cluster_vids:
                for f in by_vendor.get(cv, []):
                    if f.suppressed:
                        continue
                    if f.severity != Severity.CRITICAL:
                        f.severity = Severity.CRITICAL
                        f.description += (
                            " [Escalated: Related vendors exhibiting "
                            "coordinated overpricing]"
                        )
                        escalated_ids.append(f.finding_id)

            if escalated_ids:
                self._log(
                    "escalate", "rule6_collusion_pricing",
                    escalated_ids,
                    "Collusion cluster with pricing anomalies — "
                    "all related findings escalated to CRITICAL",
                )
            break  # only process once
