"""Vendor collusion and relationship pattern detection.

Five detection methods:

1. **Relationship network mapping** — build a vendor-to-vendor
   similarity graph by comparing addresses, phones, bank accounts,
   tax IDs, names, and contact persons.  Cluster linked vendors.
2. **Coordinated invoicing patterns** — Pearson correlation of
   invoice-date timelines between related vendor pairs.
3. **Overbilling benchmark** — compare per-vendor prices against the
   internal average and flag vendors above the 75th percentile + 15 %.
4. **Approval concentration** — flag vendors whose invoices are
   approved by a single employee.
5. **Benford's Law** — per-vendor first-digit anomaly.

Output includes network-graph data (nodes / edges) for visualisation.
"""

import logging
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz

from src.detection.base_detector import BaseDetector
from src.utils.constants import Finding, ModuleName, Severity
from src.utils.date_utils import parse_date
from src.utils.similarity import (
    normalize_address,
    normalize_phone,
    normalize_vendor_name,
)
from src.utils.statistics import benfords_test, pearson_correlation

logger = logging.getLogger(__name__)


def _to_date(val) -> Optional[date]:
    if isinstance(val, date):
        return val
    return parse_date(str(val)) if val is not None else None


def _to_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _safe(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


class VendorCollusionDetector(BaseDetector):
    """Detect vendor collusion via relationship networks and behavioural correlation."""

    def get_module_name(self) -> ModuleName:
        return ModuleName.VENDOR_COLLUSION

    def get_required_fields(self) -> list[str]:
        return ["vendor_id", "vendor_name", "total_amount",
                "invoice_date", "line_item_description"]

    def get_optional_fields(self) -> list[str]:
        return ["vendor_address", "vendor_phone", "vendor_tax_id",
                "vendor_bank_account_last4", "vendor_contact_person",
                "approved_by"]

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def detect(
        self,
        invoice_data: pd.DataFrame,
        vendor_data: Optional[pd.DataFrame] = None,
        supplementary_data: Optional[dict] = None,
    ) -> list[Finding]:
        self.findings = []
        self.stats["invoices_analyzed"] = len(invoice_data)

        vendors = self._build_vendor_list(invoice_data, vendor_data)
        inv_groups = self._group_invoices(invoice_data)

        # Method 1
        edges, clusters = self._build_relationship_network(vendors)
        self.stats["tier1_calls"] += 1

        # Method 2
        self._detect_coordinated_invoicing(clusters, inv_groups, vendors)

        # Method 3
        self._detect_overbilling(inv_groups, clusters, vendors)

        # Method 4
        self._detect_approval_concentration(invoice_data, clusters, vendors)

        # Method 5
        self._detect_benfords_anomaly(inv_groups, vendors)

        return self.findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_vendor_list(
        inv_df: pd.DataFrame, vendor_df: Optional[pd.DataFrame],
    ) -> dict[str, dict]:
        """Merge invoice-level and vendor-master data into one record per vendor."""
        vendors: dict[str, dict] = {}

        # Start with invoice data (always present)
        for _, row in inv_df.iterrows():
            vid = _safe(row.get("vendor_id"))
            if not vid or vid in vendors:
                continue
            vendors[vid] = {
                "vendor_id": vid,
                "name": _safe(row.get("vendor_name", vid)),
                "address": _safe(row.get("vendor_address", "")),
                "phone": _safe(row.get("vendor_phone", "")),
                "tax_id": _safe(row.get("vendor_tax_id", "")),
                "bank": _safe(row.get("vendor_bank_account_last4", "")),
                "contact": _safe(row.get("vendor_contact_person", "")),
            }

        # Enrich from vendor master
        if vendor_df is not None and len(vendor_df) > 0:
            for _, row in vendor_df.iterrows():
                vid = _safe(row.get("vendor_id"))
                if not vid:
                    continue
                if vid not in vendors:
                    vendors[vid] = {
                        "vendor_id": vid,
                        "name": _safe(row.get("vendor_name", vid)),
                        "address": "", "phone": "", "tax_id": "",
                        "bank": "", "contact": "",
                    }
                v = vendors[vid]
                for src, dst in [
                    ("address", "address"), ("phone", "phone"),
                    ("tax_id", "tax_id"),
                    ("bank_account_last4", "bank"),
                    ("contact_person", "contact"),
                ]:
                    val = _safe(row.get(src))
                    if val and not v[dst]:
                        v[dst] = val

        return vendors

    @staticmethod
    def _group_invoices(df: pd.DataFrame) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = defaultdict(list)
        id_col = "invoice_id" if "invoice_id" in df.columns else "invoice_number"
        for _, row in df.iterrows():
            vid = _safe(row.get("vendor_id"))
            if not vid:
                continue
            groups[vid].append({
                "invoice_id": _safe(row.get(id_col)),
                "date": _to_date(row.get("invoice_date")),
                "amount": _to_float(row.get("total_amount")),
                "description": _safe(row.get("line_item_description")),
                "approved_by": _safe(row.get("approved_by")),
            })
        return dict(groups)

    # ==================================================================
    # Method 1 — Relationship Network Mapping
    # ==================================================================

    def _build_relationship_network(
        self, vendors: dict[str, dict],
    ) -> tuple[list[dict], list[set[str]]]:
        threshold = self.config.get("relationship_composite_threshold", 0.60)

        vendor_list = list(vendors.values())
        edges: list[dict] = []

        for i in range(len(vendor_list)):
            for j in range(i + 1, len(vendor_list)):
                va, vb = vendor_list[i], vendor_list[j]
                score, signals = self._relationship_score(va, vb)
                if score >= threshold:
                    edges.append({
                        "vendor_a": va["vendor_id"],
                        "vendor_b": vb["vendor_id"],
                        "score": round(score, 3),
                        "signals": signals,
                    })

        # Build clusters via union-find
        clusters = self._cluster_edges(edges)

        # Emit findings for each relationship pair
        for edge in edges:
            va = vendors[edge["vendor_a"]]
            vb = vendors[edge["vendor_b"]]
            self.create_finding(
                severity=Severity.REVIEW,
                confidence=round(min(edge["score"], 0.95), 2),
                vendor_id=edge["vendor_a"],
                vendor_name=va["name"],
                invoice_ids=[],
                amount_at_risk=Decimal("0"),
                description=(
                    f"Vendor relationship: {va['name']} and {vb['name']} "
                    f"share attributes (score {edge['score']:.2f}): "
                    + ", ".join(edge["signals"])
                ),
                evidence={
                    "method": "relationship_network",
                    "vendor_a": edge["vendor_a"],
                    "vendor_b": edge["vendor_b"],
                    "name_a": va["name"],
                    "name_b": vb["name"],
                    "composite_score": edge["score"],
                    "signals": edge["signals"],
                },
                recommended_action=(
                    "Investigate shared vendor attributes for potential collusion"
                ),
            )

        return edges, clusters

    @staticmethod
    def _relationship_score(va: dict, vb: dict) -> tuple[float, list[str]]:
        """Weighted composite similarity between two vendors."""
        score = 0.0
        signals: list[str] = []

        # Address — different companies at the same address is a very
        # strong collusion indicator.  An exact or near-exact match
        # (>= 90%) on a substantive address is worth 0.65 alone.
        addr_a = normalize_address(va.get("address", ""))
        addr_b = normalize_address(vb.get("address", ""))
        if addr_a and addr_b and len(addr_a) >= 5:
            addr_sim = fuzz.token_sort_ratio(addr_a, addr_b) / 100.0
            if addr_sim >= 0.90:
                score += 0.65
                signals.append(f"shared address ({addr_sim:.0%})")
            elif addr_sim >= 0.75:
                score += 0.35 * addr_sim
                signals.append(f"similar address ({addr_sim:.0%})")

        # Bank — different companies sharing the same bank account last-4
        # is extremely suspicious (worth 0.65 alone).
        bank_a = va.get("bank", "")
        bank_b = vb.get("bank", "")
        if bank_a and bank_b and bank_a == bank_b:
            score += 0.65
            signals.append(f"same bank *{bank_a}")

        # Phone (weight 0.15)
        ph_a = normalize_phone(va.get("phone", ""))
        ph_b = normalize_phone(vb.get("phone", ""))
        if ph_a and ph_b and ph_a == ph_b:
            score += 0.15
            signals.append("same phone")

        # Tax ID (weight 0.15)
        tax_a = va.get("tax_id", "")
        tax_b = vb.get("tax_id", "")
        if tax_a and tax_b and tax_a == tax_b:
            score += 0.15
            signals.append("same tax ID")

        # Name similarity (weight 0.15)
        name_a = normalize_vendor_name(va.get("name", ""))
        name_b = normalize_vendor_name(vb.get("name", ""))
        if name_a and name_b:
            name_sim = fuzz.token_sort_ratio(name_a, name_b) / 100.0
            if name_sim >= 0.50:
                score += 0.15 * name_sim
                signals.append(f"similar name ({name_sim:.0%})")

        # Contact person (weight 0.10)
        ct_a = va.get("contact", "").lower().strip()
        ct_b = vb.get("contact", "").lower().strip()
        if ct_a and ct_b and ct_a == ct_b:
            score += 0.10
            signals.append(f"same contact: {va.get('contact')}")

        return score, signals

    @staticmethod
    def _cluster_edges(edges: list[dict]) -> list[set[str]]:
        """Build transitive clusters from pairwise edges (union-find)."""
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for e in edges:
            union(e["vendor_a"], e["vendor_b"])

        clusters_map: dict[str, set[str]] = defaultdict(set)
        all_nodes = set()
        for e in edges:
            all_nodes.add(e["vendor_a"])
            all_nodes.add(e["vendor_b"])
        for n in all_nodes:
            clusters_map[find(n)].add(n)

        return [c for c in clusters_map.values() if len(c) >= 2]

    # ==================================================================
    # Method 2 — Coordinated Invoicing Patterns
    # ==================================================================

    def _detect_coordinated_invoicing(
        self,
        clusters: list[set[str]],
        inv_groups: dict[str, list[dict]],
        vendors: dict[str, dict],
    ) -> None:
        timing_thresh = self.config.get("timing_correlation_threshold", 0.7)
        self.stats["tier1_calls"] += 1

        for cluster in clusters:
            vids = sorted(cluster)
            for i in range(len(vids)):
                for j in range(i + 1, len(vids)):
                    vid_a, vid_b = vids[i], vids[j]
                    invs_a = inv_groups.get(vid_a, [])
                    invs_b = inv_groups.get(vid_b, [])
                    if len(invs_a) < 3 or len(invs_b) < 3:
                        continue

                    corr = self._date_correlation(invs_a, invs_b)
                    if corr is None or corr < timing_thresh:
                        continue

                    va = vendors.get(vid_a, {})
                    vb = vendors.get(vid_b, {})
                    inv_ids = (
                        [r["invoice_id"] for r in invs_a[:5]]
                        + [r["invoice_id"] for r in invs_b[:5]]
                    )
                    total_spend = sum(r["amount"] for r in invs_a + invs_b)

                    self.create_finding(
                        severity=Severity.CRITICAL,
                        confidence=round(min(corr, 0.95), 2),
                        vendor_id=vid_a,
                        vendor_name=va.get("name", vid_a),
                        invoice_ids=inv_ids,
                        amount_at_risk=Decimal(str(round(total_spend * 0.20, 2))),
                        description=(
                            f"Coordinated invoicing: {va.get('name', vid_a)} and "
                            f"{vb.get('name', vid_b)} invoice on correlated dates "
                            f"(r={corr:.2f})"
                        ),
                        evidence={
                            "method": "coordinated_invoicing",
                            "vendor_a": vid_a,
                            "vendor_b": vid_b,
                            "correlation": round(corr, 3),
                            "cluster": sorted(cluster),
                        },
                        recommended_action=(
                            "Compare invoicing timelines side-by-side and "
                            "investigate shared-work patterns"
                        ),
                    )

    @staticmethod
    def _date_correlation(
        invs_a: list[dict], invs_b: list[dict],
    ) -> Optional[float]:
        """Pearson correlation of daily invoice counts between two vendors."""
        dates_a = [r["date"] for r in invs_a if r["date"]]
        dates_b = [r["date"] for r in invs_b if r["date"]]
        if not dates_a or not dates_b:
            return None

        all_dates = sorted(set(dates_a + dates_b))
        if len(all_dates) < 3:
            return None

        day_min = all_dates[0].toordinal()
        day_max = all_dates[-1].toordinal()
        span = day_max - day_min + 1
        if span < 30:
            return None

        # Monthly buckets
        n_buckets = max(span // 30, 2)
        counts_a = [0] * n_buckets
        counts_b = [0] * n_buckets
        for d in dates_a:
            idx = min((d.toordinal() - day_min) // 30, n_buckets - 1)
            counts_a[idx] += 1
        for d in dates_b:
            idx = min((d.toordinal() - day_min) // 30, n_buckets - 1)
            counts_b[idx] += 1

        return pearson_correlation(
            [float(c) for c in counts_a],
            [float(c) for c in counts_b],
        )

    # ==================================================================
    # Method 3 — Overbilling Benchmark
    # ==================================================================

    def _detect_overbilling(
        self,
        inv_groups: dict[str, list[dict]],
        clusters: list[set[str]],
        vendors: dict[str, dict],
    ) -> None:
        percentile = self.config.get("overbilling_percentile", 75)
        margin_pct = self.config.get("overbilling_margin_pct", 15)
        self.stats["tier1_calls"] += 1

        cluster_vids = set()
        for c in clusters:
            cluster_vids |= c

        # Compute average amount per vendor
        vendor_avgs: dict[str, float] = {}
        for vid, invs in inv_groups.items():
            amounts = [r["amount"] for r in invs if r["amount"] > 0]
            if amounts:
                vendor_avgs[vid] = sum(amounts) / len(amounts)

        if len(vendor_avgs) < 3:
            return

        import numpy as np
        all_avgs = list(vendor_avgs.values())
        p_val = float(np.percentile(all_avgs, percentile))
        threshold = p_val * (1 + margin_pct / 100)

        for vid in cluster_vids:
            avg = vendor_avgs.get(vid)
            if avg is None or avg <= threshold:
                continue

            v = vendors.get(vid, {})
            premium_pct = (avg - p_val) / p_val * 100 if p_val > 0 else 0
            invs = inv_groups.get(vid, [])
            total_spend = sum(r["amount"] for r in invs)
            inv_ids = [r["invoice_id"] for r in invs[:10]]

            self.create_finding(
                severity=Severity.REVIEW,
                confidence=0.75,
                vendor_id=vid,
                vendor_name=v.get("name", vid),
                invoice_ids=inv_ids,
                amount_at_risk=Decimal(str(round(
                    total_spend * premium_pct / (100 + premium_pct), 2,
                ))),
                description=(
                    f"Overbilling: {v.get('name', vid)} averages "
                    f"${avg:,.0f} — {premium_pct:+.0f}% above p{percentile} "
                    f"(${p_val:,.0f}) — vendor is in a relationship cluster"
                ),
                evidence={
                    "method": "overbilling",
                    "vendor_avg": round(avg, 2),
                    "percentile_value": round(p_val, 2),
                    "premium_pct": round(premium_pct, 2),
                    "total_spend": round(total_spend, 2),
                },
                recommended_action="Request competitive quotes for this vendor's services",
            )

    # ==================================================================
    # Method 4 — Approval Concentration
    # ==================================================================

    def _detect_approval_concentration(
        self,
        df: pd.DataFrame,
        clusters: list[set[str]],
        vendors: dict[str, dict],
    ) -> None:
        if "approved_by" not in df.columns:
            return

        warning_thresh = self.config.get("approval_concentration_warning", 0.3)
        self.stats["tier1_calls"] += 1

        cluster_vids = set()
        for c in clusters:
            cluster_vids |= c

        # Count approvers per vendor
        vendor_approvals: dict[str, Counter] = defaultdict(Counter)
        for _, row in df.iterrows():
            vid = _safe(row.get("vendor_id"))
            approver = _safe(row.get("approved_by"))
            if vid and approver:
                vendor_approvals[vid][approver] += 1

        for vid in cluster_vids:
            counts = vendor_approvals.get(vid)
            if not counts:
                continue
            total = sum(counts.values())
            if total < 3:
                continue
            top_approver, top_count = counts.most_common(1)[0]
            concentration = top_count / total

            if concentration < (1 - warning_thresh):
                continue  # reasonably distributed

            v = vendors.get(vid, {})
            self.create_finding(
                severity=Severity.INFORMATIONAL,
                confidence=0.65,
                vendor_id=vid,
                vendor_name=v.get("name", vid),
                invoice_ids=[],
                amount_at_risk=Decimal("0"),
                description=(
                    f"Approval concentration: {concentration:.0%} of "
                    f"{v.get('name', vid)}'s invoices approved by "
                    f"{top_approver} — vendor is in a relationship cluster"
                ),
                evidence={
                    "method": "approval_concentration",
                    "top_approver": top_approver,
                    "concentration": round(concentration, 3),
                    "total_approvals": total,
                    "approver_breakdown": dict(counts),
                },
                recommended_action="Ensure segregation of duties for flagged vendors",
            )

    # ==================================================================
    # Method 5 — Benford's Law
    # ==================================================================

    def _detect_benfords_anomaly(
        self,
        inv_groups: dict[str, list[dict]],
        vendors: dict[str, dict],
    ) -> None:
        min_invoices = self.config.get("benfords_min_invoices", 50)
        p_threshold = self.config.get("benfords_p_value", 0.05)
        self.stats["tier1_calls"] += 1

        for vid, invs in inv_groups.items():
            if len(invs) < min_invoices:
                continue

            amounts = [Decimal(str(r["amount"])) for r in invs if r["amount"] > 0]
            if len(amounts) < min_invoices:
                continue

            result = benfords_test(amounts, digits=1)
            if result["significant"] and result["p_value"] < p_threshold:
                v = vendors.get(vid, {})
                self.create_finding(
                    severity=Severity.INFORMATIONAL,
                    confidence=0.60,
                    vendor_id=vid,
                    vendor_name=v.get("name", vid),
                    invoice_ids=[r["invoice_id"] for r in invs[:10]],
                    amount_at_risk=Decimal("0"),
                    description=(
                        f"Benford's anomaly for {v.get('name', vid)}: "
                        f"MAD={result['mad']:.4f}, p={result['p_value']:.4f}"
                    ),
                    evidence={
                        "method": "benfords",
                        "chi_squared": result["chi_squared"],
                        "p_value": result["p_value"],
                        "mad": result["mad"],
                        "sample_size": len(amounts),
                    },
                    recommended_action="Investigate invoice amount distribution",
                )
