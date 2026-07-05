"""Shared test fixtures for SilentAuditor.

Provides reusable fixtures for caches, configs, mock LLM clients,
and pre-built DataFrames at various quality levels.
"""

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

from src.normalization.cache import CacheManager


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DATA_DIR = _PROJECT_ROOT / "data"


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@pytest.fixture
def sample_config() -> dict:
    """Full config loaded from thresholds.yaml."""
    with open(_CONFIG_DIR / "thresholds.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Cache
# ------------------------------------------------------------------

@pytest.fixture
def mock_cache(tmp_path) -> CacheManager:
    """In-memory-style SQLite cache backed by a temp directory."""
    cm = CacheManager(db_path=str(tmp_path / "test_cache.db"))
    yield cm
    cm.close()


# ------------------------------------------------------------------
# Mock LLM client
# ------------------------------------------------------------------

@pytest.fixture
def mock_llm_client() -> MagicMock:
    """LLM client with all API calls mocked (no network access).

    * ``classify_by_keywords`` delegates to a real keyword-match
      implementation so Tier-1 tests behave realistically.
    * ``compute_text_similarity`` returns a deterministic score
      based on string overlap.
    * ``assess_duplicate_pair`` defaults to *not duplicate*.
    * ``classify_line_item`` returns ``Uncategorized``.
    * Embedding methods return deterministic 384-dim vectors.
    """
    m = MagicMock()

    # ---- Tier 1: keyword classification (real logic) ------------------
    def _kw_classify(desc, kw_map):
        if not desc or not kw_map:
            return None
        desc_lower = desc.lower()
        best, best_len = None, 0
        for kw, cat in kw_map.items():
            if kw.lower() in desc_lower and len(kw) > best_len:
                best, best_len = cat, len(kw)
        return best

    m.classify_by_keywords.side_effect = _kw_classify

    # ---- Tier 2: deterministic embeddings ----------------------------
    def _text_sim(a, b):
        if a == b:
            return 1.0
        sa, sb = set(a.lower().split()), set(b.lower().split())
        union = sa | sb
        if not union:
            return 0.0
        return len(sa & sb) / len(union)

    m.compute_text_similarity.side_effect = _text_sim

    def _get_embedding(text):
        rng = np.random.default_rng(hash(text) % 2**32)
        return rng.random(384).astype(np.float32)

    m.get_embedding.side_effect = _get_embedding

    # ---- Tier 3: LLM stubs ------------------------------------------
    m.assess_duplicate_pair.return_value = {
        "is_duplicate": False,
        "confidence": 0.3,
        "reasoning": "mock",
    }
    m.classify_line_item.return_value = {
        "category": "Uncategorized",
        "subcategory": None,
        "item_type": None,
        "confidence": 0.70,
        "source": "llm",
    }
    m.score_description_vagueness.return_value = 50

    return m


# ------------------------------------------------------------------
# Sample DataFrames
# ------------------------------------------------------------------

@pytest.fixture
def sample_invoice_df() -> pd.DataFrame:
    """100 clean invoices from 10 vendors, no fraud.

    18-month span, realistic amounts, POs on 60% of invoices.
    """
    rng = np.random.default_rng(100)
    rows = []
    vendors = [(f"V{i:03d}", f"Vendor {i}") for i in range(10)]

    for i in range(100):
        vid, vname = vendors[i % 10]
        month = (i % 18) + 1
        year = 2024 + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        inv_date = date(year, month, int(rng.integers(1, 28)))
        amt = round(float(rng.uniform(500, 20000)), 2)
        has_po = rng.random() < 0.6
        rows.append({
            "invoice_id": f"INV-{i:04d}",
            "invoice_number": f"INV-{i:04d}",
            "vendor_id": vid,
            "vendor_name": vname,
            "invoice_date": inv_date,
            "total_amount": amt,
            "po_number": f"PO-{rng.integers(10000,99999)}" if has_po else "",
            "line_item_description": f"Service item {i}",
            "unit_price": round(amt / max(int(rng.integers(1, 5)), 1), 2),
            "quantity": int(rng.integers(1, 10)),
            "payment_date": inv_date + timedelta(days=int(rng.integers(20, 50))),
            "payment_status": "Paid",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def fraud_invoice_df() -> pd.DataFrame:
    """100 invoices with 4 known fraud patterns injected.

    Fraud items (indices given for validation):
    * Exact duplicate: rows 90-91 (same inv number/vendor/amount)
    * Near duplicate: rows 92-93 (amount within 1%, 3 days apart)
    * Price creep: vendor V001 rows 0,10,20,30,40,50 (steady increase)
    * Split invoicing: rows 94-97 ($4,900 each → $19,600 > $5K threshold)
    """
    rng = np.random.default_rng(200)
    rows = []

    # 90 clean invoices from 10 vendors
    for i in range(90):
        vid = f"V{i % 10:03d}"
        month = (i % 12) + 1
        base_price = 100 + (i % 10) * 20
        # Inject price creep on V001 (indices 1,11,21,31,41,51,61,71,81)
        if vid == "V001":
            base_price = 100 + (i // 10) * 12  # +12% per set of 10
        rows.append({
            "invoice_id": f"INV-{i:04d}",
            "invoice_number": f"INV-{i:04d}",
            "vendor_id": vid,
            "vendor_name": f"Vendor {vid}",
            "invoice_date": date(2024, month, 15),
            "total_amount": round(base_price * float(rng.uniform(8, 12)), 2),
            "po_number": f"PO-{rng.integers(10000,99999)}",
            "line_item_description": f"Standard service item {vid}",
            "unit_price": float(base_price),
            "quantity": int(rng.integers(8, 12)),
        })

    # Exact duplicate (rows 90, 91)
    rows.append({**rows[5], "invoice_id": "INV-0005", "invoice_number": "INV-0005"})

    # Near duplicate (rows 92, 93)
    rows.append({
        "invoice_id": "INV-0092", "invoice_number": "INV-0092",
        "vendor_id": "V005", "vendor_name": "Vendor V005",
        "invoice_date": date(2024, 3, 10),
        "total_amount": 8000.00,
        "line_item_description": "Consulting engagement Q1",
        "unit_price": 200.0, "quantity": 40,
        "po_number": "PO-50000",
    })
    rows.append({
        "invoice_id": "INV-0093", "invoice_number": "INV-0093",
        "vendor_id": "V005", "vendor_name": "Vendor V005",
        "invoice_date": date(2024, 3, 13),
        "total_amount": 8050.00,
        "line_item_description": "Consulting engagement Q1",
        "unit_price": 201.25, "quantity": 40,
        "po_number": "PO-50001",
    })

    # Split invoicing (rows 94-97): 4 × $4900 same PO
    for j in range(4):
        rows.append({
            "invoice_id": f"INV-009{4+j}",
            "invoice_number": f"INV-009{4+j}",
            "vendor_id": "V009",
            "vendor_name": "Vendor V009",
            "invoice_date": date(2025, 1, 5 + j * 2),
            "total_amount": 4900.00,
            "po_number": "PO-SPLIT",
            "line_item_description": "Equipment and supplies",
            "unit_price": 4900.0, "quantity": 1,
        })

    return pd.DataFrame(rows)


@pytest.fixture
def messy_invoice_df() -> pd.DataFrame:
    """100 invoices with messy data quality.

    Issues: unparseable dates, non-numeric amounts, missing vendor names,
    currency symbols, whitespace, mixed date formats.
    """
    rng = np.random.default_rng(300)
    rows = []
    date_formats = [
        "01/15/2025", "2025-02-20", "15-Mar-2025", "April 10, 2025",
        "not-a-date", "", "05/30/2025", "2025-06-15",
    ]

    for i in range(100):
        vid = f"V{i % 8:03d}"
        amt_raw = rng.choice([
            "$1,500.00", "2000.50", "€3.000,50", "(100.00)",
            "abc", "", "7500", "$4,999.99",
        ])
        rows.append({
            "invoice_id": f"MESSY-{i:04d}",
            "invoice_number": f"  MESSY-{i:04d}  ",  # whitespace
            "vendor_id": vid if i % 7 != 0 else "",  # some missing
            "vendor_name": f"  Vendor {vid}  " if i % 5 != 0 else "",
            "invoice_date": date_formats[i % len(date_formats)],
            "total_amount": amt_raw,
            "po_number": f"PO-{i}" if i % 3 == 0 else "",
            "line_item_description": f"Item {i}" if i % 4 != 0 else "",
            "unit_price": str(rng.choice([100, 200, "N/A", ""])),
            "quantity": str(rng.choice([1, 5, 10, "", "N/A"])),
        })

    return pd.DataFrame(rows)
