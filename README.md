# SilentAuditor

AI-powered accounts payable fraud detection and audit system for mid-market companies. SilentAuditor ingests invoice data from any ERP system, normalises vendor and item information, then runs eight parallel detection modules to identify duplicate payments, price creep, phantom services, vendor collusion, contract violations, behavioural anomalies, market-price overcharges, and split invoicing. Findings are adjudicated, risk-scored, and delivered as actionable reports with estimated dollar recovery.

**Target users:** CFOs, Controllers, and AP managers at companies with $5M-$100M in annual revenue, particularly in construction, manufacturing, and healthcare where AP controls are weakest and recoverable leakage averages 1-3% of vendor spend.

---

## Detection Capabilities

| # | Module | What It Detects |
|---|--------|----------------|
| 1 | Duplicate Invoices | Exact, near-duplicate, reformatted, and cross-PO duplicate payments |
| 2 | Price Creep | Gradual increases, sudden jumps, accelerating trends, contract-rate breaches |
| 3 | Phantom Services | Vague descriptions, vendor-category mismatches, missing receipts |
| 4 | Vendor Collusion | Shared addresses/banks, coordinated invoicing, overbilling clusters |
| 5 | Contract Compliance | Rate overcharges, missed discounts, out-of-scope billing, expired contracts |
| 6 | Vendor Behavior | Bank account changes (BEC), amount spikes, frequency shifts, identity changes |
| 7 | Market Price | Above-market pricing, vendor premium indices, renegotiation opportunities |
| 8 | Split Invoicing | Threshold clustering, temporal splitting, PO-linked splits, Benford's anomalies |

---

## Quick Start

### Installation

```bash
# Clone and install
cd silentauditor
pip install -r requirements.txt
pip install -e .
```

### Generate Sample Data

```bash
silentauditor generate-sample-data --output-dir data/ --seed 42
```

This creates 7,500+ invoice rows across 141 vendors with 56 injected fraud patterns spanning all 7 fraud types, plus vendor master data, contracts, goods receipts, approval thresholds, and a ground-truth fraud key.

### Run Analysis

```bash
# Full analysis with all data sources
silentauditor analyze \
  -i data/sample_invoices/invoices.csv \
  -v data/sample_vendor_master/vendors.csv \
  -c data/sample_contracts/contracts.json \
  -a data/sample_invoices/approval_thresholds.json \
  -r data/sample_invoices/goods_receipts.csv \
  -o results/

# Minimal run (invoice data only)
silentauditor analyze -i invoices.csv -o results/

# Skip specific modules
silentauditor analyze -i invoices.csv --skip-modules vendor_collusion --skip-modules market_price

# Use ERP template for column mapping
silentauditor analyze -i netsuite_export.csv --erp netsuite -o results/

# Verbose logging
silentauditor analyze -i invoices.csv -o results/ --verbose
```

### View Results

```
results/
├── executive_summary.json      # Leadership overview
├── executive_summary.html      # Formatted HTML report
├── detailed_findings.json      # Every finding with evidence
├── detailed_findings.csv       # Spreadsheet-friendly format
├── vendor_risk_report.json     # Per-vendor risk scores
├── vendor_risk_report.html     # Formatted vendor report
├── recovery_opportunities.json # Dollar recovery estimates
├── recovery_opportunities.csv  # Spreadsheet format
└── dashboard_data.json         # Single JSON for frontend dashboards
```

---

## Input Data Requirements

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `invoice_number` | string | Unique invoice identifier |
| `vendor_name` | string | Vendor/supplier name |
| `invoice_date` | date | Invoice issue date |
| `total_amount` | decimal | Invoice total amount |

### Optional Fields (Enhance Detection)

| Field | Type | Enhances |
|-------|------|----------|
| `vendor_id` | string | All modules (vendor matching) |
| `line_item_description` | string | Modules 2, 3, 5, 7 |
| `unit_price` | decimal | Modules 2, 5, 7 |
| `quantity` | decimal | Modules 2, 5, 7, 8 |
| `po_number` | string | Modules 1, 3, 8 |
| `payment_date` | date | Module 5 (payment terms) |
| `payment_status` | string | General context |
| `approved_by` | string | Module 4 (approval concentration) |

### Supplementary Data

| Data Source | Format | Used By |
|------------|--------|---------|
| Vendor Master | CSV | Modules 4, 6 (relationship network, baseline) |
| Contracts | JSON | Module 5 (rate/scope/expiry checks) |
| Goods Receipts | CSV | Module 3 (delivery verification) |
| Approval Thresholds | JSON | Module 8 (split detection) |

### Supported Formats

- **CSV** (auto-detects encoding: UTF-8, Latin-1, CP1252; delimiters: comma, semicolon, tab, pipe)
- **Excel** (.xlsx, .xls via openpyxl)
- **JSON** (array-of-objects or nested structures)
- **TSV** (tab-separated)

### ERP Templates

Pre-built column mappings for: `netsuite`, `quickbooks`, `sage`, `sap_b1`, `xero`. Use `--erp <name>` to apply. Templates are in `config/erp_templates/`.

---

## Detection Modules

### Module 1: Duplicate Invoice Detection

Detects payments made twice for the same work using four layers. **Layer 1** catches exact matches (same invoice number, vendor, amount). **Layer 2** uses fuzzy composite scoring (amount similarity, date proximity, line-item embeddings, PO matching) to find near-duplicates. **Layer 3** sends ambiguous pairs to the LLM for semantic analysis. **Layer 4** finds cross-PO duplicates where the same work is billed under different purchase orders. False-positive suppression removes recurring subscriptions, progress billing, and credit/rebill pairs.

### Module 2: Price Creep Detection

Builds chronological price time series for each vendor-item pair and runs statistical trend analysis. Detects gradual cumulative drift (exceeding 8% over 6 months or 12% over 12 months), sudden single-price jumps (>5%), and accelerating increase patterns. Compares current prices against contracted rates with escalation clauses. Volume changes are noted as context that may explain price movement.

### Module 3: Phantom Services Detection

Identifies invoices for goods or services that may not have been delivered. Scores line-item descriptions for vagueness, flags vendor-category mismatches (e.g., an office supply vendor billing for engineering consulting), detects one-off charges outside a vendor's normal scope, matches invoices against PO records, and cross-references material invoices against goods receipts. Runs in degraded mode when PO/receipt data is unavailable.

### Module 4: Vendor Collusion Detection

Maps vendor relationships by comparing addresses, bank accounts, phone numbers, tax IDs, names, and contacts. Builds clusters using transitive relationships (union-find). Analyses coordinated invoicing patterns via date correlation, benchmarks cluster-vendor pricing against internal averages, checks approval concentration, and runs Benford's Law analysis per vendor.

### Module 5: Contract Compliance Verification

Verifies invoiced charges against contract terms across six dimensions: rate compliance (with escalation), payment term compliance (missed early-payment discounts), volume commitment verification (unapplied quantity discounts), scope boundary monitoring (line items outside contract scope), contract expiration alerts, and escalation clause verification. Requires contract data to run.

### Module 6: Vendor Behavior Anomaly Detection

Builds an 8-dimension behavioural baseline for each vendor (frequency, amount distribution, timing, categories, contacts, bank accounts, email, invoice format). Scores recent invoices against the baseline. Bank account changes trigger immediate CRITICAL alerts (potential BEC attack). Multi-anomaly compound alerts escalate when 3+ dimensions shift simultaneously. Drift detection compares current and prior baselines to catch gradual behavioural shifts.

### Module 7: Market Price Benchmarking

Constructs internal price benchmarks by category across all vendors (median, p25, p75, p90). Flags individual items priced above p75 + 15%. Computes a per-vendor premium index and flags vendors with weighted-average premium >15%. Estimates renegotiation savings if vendor prices were brought to category median. Framework ready for cross-client anonymised benchmark data.

### Module 8: Split Invoicing Detection

Detects approval-threshold circumvention using five methods: statistical clustering analysis below each threshold, sliding-window temporal aggregation, PO/project-linked split detection, per-vendor Benford's Law analysis, and historical pattern comparison (average-amount drop with maintained spend or frequency increase without spend increase).

---

## Configuration

### Thresholds

All configurable parameters live in `config/thresholds.yaml`. Key settings:

```yaml
duplicate_detection:
  fuzzy_composite_threshold: 0.85   # Lower = more sensitive, more FPs
  date_window_days: 90              # Comparison window for fuzzy matching

price_creep:
  cumulative_threshold_6mo_pct: 8   # Flag cumulative increase > 8% over 6 months
  single_jump_threshold_pct: 5      # Flag single price jump > 5%

phantom_services:
  vagueness_flag_threshold: 40      # Specificity score below this gets flagged (0-100)
  category_mismatch_min_amount: 1000

split_invoicing:
  default_approval_thresholds: [5000, 25000, 100000]
  below_threshold_band_pct: 10      # % band below threshold to check for clustering
```

### LLM Settings

Set in `config/settings.py`:

```python
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_MONTHLY_COST_PER_CUSTOMER = 200.0  # USD cap
EMBEDDING_MODEL = "all-MiniLM-L6-v2"       # Local model, no API cost
```

Set your API key as an environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

If no API key is set, the system runs with Tier 1 (rules) and Tier 2 (local embeddings) only. Tier 3 (Claude API) is skipped gracefully.

---

## Output & Reports

### Executive Summary

JSON + HTML overview for leadership. Includes severity breakdown, total amount at risk, top 5 risk vendors, recovery opportunities, per-module summary, and data quality overview.

### Detailed Findings

JSON + CSV with every active finding. Each entry includes: finding ID, module, severity, confidence, vendor, invoice IDs, amount at risk, description, evidence, and recommended action. Sorted by severity then amount.

### Vendor Risk Report

JSON + HTML per-vendor risk profiles. Each vendor has a 0-100 risk score, risk tier (CRITICAL/HIGH/MEDIUM/LOW), contributing factors, and all related findings. Sorted by risk score descending.

### Recovery Opportunities

JSON + CSV of high-confidence findings sorted by amount. Includes gain-share calculation (25% of recoverable amount).

### Dashboard Data

Single JSON blob with all data needed for a frontend dashboard: summary stats, risk distribution, findings by module, amount by module, top vendors, top findings, timeline trend data, and data quality metrics.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUT FILES                                  │
│            CSV / Excel / JSON from any ERP system                    │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐
│     INGESTION LAYER      │  FileLoader → FieldMapper → DataQualityAssessor
│  Auto-detect columns     │  ERP templates: NetSuite, QuickBooks, Sage,
│  Parse dates & amounts   │  SAP B1, Xero
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   NORMALIZATION LAYER    │  VendorNormalizer (fuzzy dedup)
│  SQLite-cached results   │  ItemNormalizer (tiered classification)
│  Tiered LLM processing   │  AmountNormalizer (currency/tax/credit)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│              DETECTION MODULES (8 parallel modules)                  │
│                                                                      │
│  1. Duplicates    3. Phantom     5. Contract    7. Market Price      │
│  2. Price Creep   4. Collusion   6. Behavior    8. Split Invoicing   │
│                                                                      │
│  All inherit from BaseDetector → consistent interface                │
│  ModuleRunner orchestrates execution, handles failures               │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│   ADJUDICATION ENGINE    │  ConflictResolver (6 rules)
│  Resolve contradictions  │  FalsePositiveManager (feedback learning)
│  Score & prioritise      │  RiskScorer (vendor + invoice level)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   REPORTING & OUTPUT     │  AlertManager (critical/review/informational)
│  JSON, CSV, HTML         │  ReportGenerator (4 report types)
│  Dashboard-ready data    │  DashboardDataGenerator
└──────────────────────────┘
```

### Tiered LLM Processing

Cost is controlled via a three-tier architecture:

- **Tier 1 — Rule-based (zero cost):** Keyword matching, regex, statistical tests, Benford's Law, z-scores. Handles 60-70% of all detection logic.
- **Tier 2 — Local embeddings (minimal cost):** sentence-transformers model (`all-MiniLM-L6-v2`) running locally. Used for text similarity, item classification, duplicate description matching. All embeddings cached in SQLite.
- **Tier 3 — Claude API (expensive):** Only for ambiguous cases that Tier 1 and 2 cannot resolve. Responses cached. Monthly budget cap per customer ($200 default).

### Caching Strategy

SQLite database (`~/.silentauditor/cache.db`) stores:
- Vendor name normalisation mappings
- Item classification results
- Text embeddings (computed once per unique text)
- LLM prompt/response pairs (identical questions never hit the API twice)
- User feedback for false-positive learning
- Vendor behavioural baselines

---

## Feedback & Learning

### Recording Feedback

```bash
# Mark a finding as false positive
silentauditor feedback --finding-id SA-abc123def456 --verdict false_positive --notes "Recurring subscription"

# Confirm a finding as fraud
silentauditor feedback --finding-id SA-abc123def456 --verdict confirmed_fraud

# Mark as legitimate
silentauditor feedback --finding-id SA-abc123def456 --verdict legitimate
```

### How Learning Works

When the same pattern (module + vendor + finding type) is marked `false_positive` 3+ times by a customer, future occurrences are auto-suppressed. Cross-customer learning kicks in when 3+ different customers mark the same pattern. The system also suggests threshold adjustments for modules with FP rates above 20%.

---

## Testing

### Run All Tests

```bash
# Full suite (769 tests)
pytest tests/ -v

# Quick run
pytest tests/ -q

# Specific module
pytest tests/test_duplicate_invoices.py -v
```

### Validate Detection Accuracy

```bash
# Generate sample data and validate
silentauditor generate-sample-data --seed 42
silentauditor validate -i data/ -k data/sample_invoices/fraud_key.json -o validation_results/
```

### Benchmark Performance

```bash
python scripts/benchmark_performance.py
```

Expected output: ~37s total time, ~53MB peak memory for 7,500 invoices.

### Interpreting Validation Results

The validation report shows per-fraud-type detection rates:

```
  contract_compliance             10/10   (100%)
  duplicate_invoice               15/15   (100%)
  phantom_services                12/12   (100%)
  price_creep                      8/8    (100%)
  split_invoicing                  4/4    (100%)
  vendor_behavior                  5/5    (100%)
  vendor_collusion                 2/2    (100%)
  OVERALL                         56/56   (100%)
```

---

## Troubleshooting

### Missing API Key

```
ANTHROPIC_API_KEY environment variable is not set
```

**Solution:** The system continues with Tier 1 (rules) and Tier 2 (embeddings) only. Set the key to enable Tier 3 LLM analysis for ambiguous cases:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Poor Detection Rates

**Data quality checklist:**

1. Run `silentauditor quality-check -i your_data.csv` first
2. Ensure `invoice_number`, `vendor_name`, `invoice_date`, `total_amount` are present and populated
3. Add `line_item_description` and `unit_price` for price creep and phantom service detection
4. Add `po_number` for duplicate and phantom detection
5. Provide vendor master data for collusion detection
6. Provide contracts for compliance checking
7. Ensure at least 6 months of history for vendor behaviour analysis

### High False Positive Rates

**Threshold tuning guide:**

| Problem | Adjustment |
|---------|------------|
| Too many duplicate flags | Increase `fuzzy_composite_threshold` (0.85 → 0.90) |
| Too many price creep flags | Increase `cumulative_threshold_6mo_pct` (8 → 12) |
| Too many phantom service flags | Decrease `vagueness_flag_threshold` (40 → 25) |
| Too many split invoicing flags | Increase `min_invoices_in_cluster` (3 → 5) |

Use the feedback system to teach the system your patterns:

```bash
silentauditor feedback --finding-id SA-xxx --verdict false_positive --notes "Normal pattern"
```

### Module Crashes

Individual module failures are caught and logged — other modules continue running. Check the log for `ERROR` messages. Common causes:
- Insufficient data for a module (need minimum row counts)
- Unexpected data types (ensure numeric columns are numeric)
- Memory issues on very large datasets (>100K rows)

---

## License

Proprietary. All rights reserved.
