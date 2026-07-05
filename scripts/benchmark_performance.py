#!/usr/bin/env python3
"""Benchmark SilentAuditor processing time, memory, and cache performance.

Usage::

    python scripts/benchmark_performance.py
    python scripts/benchmark_performance.py --scale 2   # 2x data (15K rows)
    python scripts/benchmark_performance.py --scale 5   # 5x data (37K rows)
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.WARNING)

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"


def _mock_llm() -> MagicMock:
    m = MagicMock()
    m.compute_text_similarity.return_value = 0.5
    m.assess_duplicate_pair.return_value = {
        "is_duplicate": False, "confidence": 0.3, "reasoning": "mock"}
    m.score_description_vagueness.return_value = 55
    return m


def _scale_data(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Duplicate the DataFrame *factor* times with unique invoice IDs."""
    if factor <= 1:
        return df
    parts = [df]
    for i in range(1, factor):
        dup = df.copy()
        dup["invoice_id"] = dup["invoice_id"].astype(str) + f"_s{i}"
        dup["invoice_number"] = dup["invoice_number"].astype(str) + f"_s{i}"
        parts.append(dup)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=1,
                        help="Scale factor for data size (1=7.5K, 2=15K, etc.)")
    args = parser.parse_args()

    inv_path = _DATA / "sample_invoices" / "invoices.csv"
    if not inv_path.exists():
        import subprocess
        subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "generate_sample_data.py"),
             "--output-dir", str(_DATA), "--seed", "42"],
            check=True, capture_output=True,
        )

    sys.path.insert(0, str(_ROOT))

    from src.normalization.cache import CacheManager
    from src.ingestion.file_loader import FileLoader
    from src.detection.base_detector import ModuleRunner
    from src.detection.duplicate_invoices import DuplicateInvoiceDetector
    from src.detection.price_creep import PriceCreepDetector
    from src.detection.phantom_services import PhantomServicesDetector
    from src.detection.vendor_collusion import VendorCollusionDetector
    from src.detection.contract_compliance import ContractComplianceDetector
    from src.detection.vendor_behavior import VendorBehaviorDetector
    from src.detection.market_price import MarketPriceDetector
    from src.detection.split_invoicing import SplitInvoiceDetector
    from src.adjudication.conflict_resolver import ConflictResolver
    from src.adjudication.risk_scorer import RiskScorer
    from src.reporting.alert_manager import AlertManager

    with open(_ROOT / "config" / "thresholds.yaml") as f:
        config = yaml.safe_load(f)

    loader = FileLoader()
    inv_df = loader.load(str(inv_path))
    vend_df = loader.load(str(_DATA / "sample_vendor_master" / "vendors.csv"))
    for c in ["total_amount", "unit_price", "quantity"]:
        if c in inv_df.columns:
            inv_df[c] = pd.to_numeric(inv_df[c], errors="coerce")
    if "invoice_id" not in inv_df.columns:
        inv_df["invoice_id"] = inv_df["invoice_number"]

    # Scale data
    inv_df = _scale_data(inv_df, args.scale)

    with open(_DATA / "sample_contracts" / "contracts.json") as f:
        contracts = json.load(f)
    with open(_DATA / "sample_invoices" / "approval_thresholds.json") as f:
        thresholds = json.load(f)
    receipts_path = _DATA / "sample_invoices" / "goods_receipts.csv"
    receipts = loader.load(str(receipts_path)) if receipts_path.exists() else None

    supp = {"contracts": contracts, "approval_thresholds": thresholds}
    if receipts is not None:
        supp["receipts"] = receipts

    db_path = os.path.join(tempfile.mkdtemp(), "bench.db")
    cache = CacheManager(db_path=db_path)
    llm = _mock_llm()

    # ---- Run 1: cold cache ----
    tracemalloc.start()
    t0 = time.perf_counter()

    runner = ModuleRunner(config, cache, llm)
    for cls in [DuplicateInvoiceDetector, PriceCreepDetector,
                PhantomServicesDetector, VendorCollusionDetector,
                ContractComplianceDetector, VendorBehaviorDetector,
                MarketPriceDetector, SplitInvoiceDetector]:
        runner.register_module(cls)

    findings = runner.run_all(inv_df, vendor_data=vend_df, supplementary_data=supp)
    findings = ConflictResolver(config).resolve(findings)
    scorer = RiskScorer()
    scorer.score_vendors(findings)
    scorer.score_invoices(findings)
    AlertManager(config).categorize_alerts(findings)

    run1_time = time.perf_counter() - t0
    _, peak1 = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats = runner.get_all_stats()
    active = [f for f in findings if not f.suppressed]
    cache_stats = cache.get_cache_stats()

    # ---- Run 2: warm cache ----
    runner2 = ModuleRunner(config, cache, llm)
    for cls in [DuplicateInvoiceDetector, PriceCreepDetector,
                PhantomServicesDetector, VendorCollusionDetector,
                ContractComplianceDetector, VendorBehaviorDetector,
                MarketPriceDetector, SplitInvoiceDetector]:
        runner2.register_module(cls)

    t0 = time.perf_counter()
    findings2 = runner2.run_all(inv_df, vendor_data=vend_df, supplementary_data=supp)
    run2_time = time.perf_counter() - t0
    cache_stats2 = cache.get_cache_stats()

    # ---- Report ----
    print("\n" + "=" * 70)
    print("  SilentAuditor — Performance Benchmark")
    print("=" * 70)
    print(f"  Data scale:        {args.scale}x ({len(inv_df):,} rows, "
          f"{inv_df['vendor_id'].nunique()} vendors)")
    print(f"  Findings:          {len(active):,} active, "
          f"{len(findings)-len(active):,} suppressed")
    print(f"  Run 1 (cold):      {run1_time:>8.2f}s")
    print(f"  Run 2 (warm):      {run2_time:>8.2f}s")
    cache_speedup = ((run1_time - run2_time) / run1_time * 100
                     if run1_time > 0 else 0)
    print(f"  Cache speedup:     {cache_speedup:>7.0f}%")
    print(f"  Peak memory:       {peak1 / 1024 / 1024:>8.1f} MB")
    print("-" * 70)

    print("  Per-Module (Run 1):")
    for ms in stats["modules"]:
        print(f"    {ms['module']:<28s}  {ms['processing_time_seconds']:>6.2f}s  "
              f"  findings={ms['findings_generated']}")

    print("-" * 70)
    print("  Cache Stats:")
    print(f"    Total cached rows:   {cache_stats2.get('total_rows', 0):,}")
    print(f"    Database size:       {cache_stats2.get('database_size_bytes', 0) / 1024:.0f} KB")
    for table, count in cache_stats2.get("tables", {}).items():
        if count > 0:
            print(f"      {table}: {count:,}")

    print("-" * 70)

    # Assertions
    ok = True
    max_time = {1: 300, 2: 600, 3: 600, 5: 1800, 7: 1800, 10: 1800}
    limit = max_time.get(args.scale, args.scale * 300)
    if run1_time > limit:
        print(f"  FAIL: run time {run1_time:.0f}s > {limit}s limit")
        ok = False
    else:
        print(f"  PASS: run time {run1_time:.1f}s < {limit}s")

    peak_gb = peak1 / 1024 / 1024 / 1024
    mem_limit = 2.0
    if peak_gb > mem_limit:
        print(f"  FAIL: peak memory {peak_gb:.2f} GB > {mem_limit} GB")
        ok = False
    else:
        print(f"  PASS: peak memory {peak1 / 1024 / 1024:.0f} MB < {mem_limit} GB")

    if run2_time < run1_time * 0.95:
        print(f"  PASS: warm cache faster ({run2_time:.1f}s vs {run1_time:.1f}s)")
    else:
        print(f"  INFO: warm cache similar ({run2_time:.1f}s vs {run1_time:.1f}s)")

    print("=" * 70)
    cache.close()

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
