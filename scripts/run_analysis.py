#!/usr/bin/env python3
"""Convenience script to run SilentAuditor analysis on sample data.

Usage::

    python scripts/run_analysis.py
    python scripts/run_analysis.py --output-dir custom_results/
"""

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run SilentAuditor on sample data")
    parser.add_argument("--output-dir", "-o", default="results")
    args = parser.parse_args()

    inv = _DATA / "sample_invoices" / "invoices.csv"
    if not inv.exists():
        print("Sample data not found — generating...")
        subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "generate_sample_data.py"),
             "--output-dir", str(_DATA), "--seed", "42"],
            check=True,
        )

    cmd = [
        sys.executable, "-m", "src.main", "analyze",
        "-i", str(_DATA / "sample_invoices" / "invoices.csv"),
        "-v", str(_DATA / "sample_vendor_master" / "vendors.csv"),
        "-c", str(_DATA / "sample_contracts" / "contracts.json"),
        "-a", str(_DATA / "sample_invoices" / "approval_thresholds.json"),
        "-r", str(_DATA / "sample_invoices" / "goods_receipts.csv"),
        "-o", args.output_dir,
    ]
    print(f"Running: {' '.join(cmd[-8:])}")
    subprocess.run(cmd, cwd=str(_ROOT), check=True)


if __name__ == "__main__":
    main()
