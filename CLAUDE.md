# CLAUDE.md — SilentAuditor Project Context

## What Is This Project

SilentAuditor is an AI-powered Accounts Payable (AP) audit system that detects invoice fraud, vendor anomalies, and financial leakage for mid-market companies ($5M–$100M revenue). It targets construction, manufacturing, and healthcare verticals where AP controls are weakest and recoverable leakage averages 1–3% of vendor spend ($80K–$750K annually per customer).

The business model is gain-share: zero upfront cost, SilentAuditor takes 25% of verified recoveries. This means the system MUST produce high-precision findings (low false positives) because every false flag wastes customer trust and every missed fraud is lost revenue.

---

## Architecture Overview

```
Input Files (CSV/Excel/JSON)
        │
        ▼
┌─────────────────────┐
│  INGESTION LAYER    │  file_loader → field_mapper → data_quality
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  NORMALIZATION      │  vendor_normalizer → item_normalizer → amount_normalizer
│  LAYER              │  (SQLite-cached, tiered LLM processing)
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  DETECTION MODULES (8 parallel modules, all inherit BaseDetector) │
│                                                                   │
│  Module 1: Duplicate Invoices      Module 5: Contract Compliance  │
│  Module 2: Price Creep             Module 6: Vendor Behavior      │
│  Module 3: Phantom Services        Module 7: Market Price         │
│  Module 4: Vendor Collusion        Module 8: Split Invoicing      │
└────────┬──────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  ADJUDICATION       │  conflict_resolver → risk_scorer → false_positive_manager
│  ENGINE             │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  REPORTING &        │  alert_manager → report_generator → dashboard_data
│  OUTPUT             │
└─────────────────────┘
```

---

## Project Structure

```
silentauditor/
├── CLAUDE.md                      ← You are here
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── settings.py                # Global config constants
│   ├── thresholds.yaml            # ALL configurable detection thresholds
│   └── erp_templates/             # Column mappings per ERP system
│       ├── netsuite.yaml
│       ├── quickbooks.yaml
│       ├── sage.yaml
│       ├── sap_b1.yaml
│       └── xero.yaml
├── data/
│   ├── sample_invoices/           # Generated test data lives here
│   ├── sample_contracts/
│   ├── sample_vendor_master/
│   └── benchmarks/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Click CLI — entry point for everything
│   ├── ingestion/                 # Data loading, field mapping, quality checks
│   ├── normalization/             # Vendor/item/amount normalization + cache
│   ├── detection/                 # 8 detection modules + base class
│   ├── adjudication/              # Cross-module conflict resolution + scoring
│   ├── reporting/                 # Reports, alerts, dashboard JSON
│   └── utils/                     # LLM client, statistics, similarity, constants
├── tests/
│   ├── conftest.py                # Shared fixtures
│   └── test_*.py                  # One test file per module
└── scripts/
    ├── generate_sample_data.py    # Generates test data with injected fraud
    ├── run_analysis.py            # Convenience runner
    └── benchmark_performance.py   # Performance benchmarking
```

---

## Tech Stack & Dependencies

- **Python 3.11+** — required minimum
- **pandas / numpy** — all data processing
- **scikit-learn** — IsolationForest for anomaly detection
- **scipy** — statistical tests (chi-squared, linear regression, Benford's)
- **rapidfuzz** — fuzzy string matching (NOT fuzzywuzzy — rapidfuzz is faster)
- **sentence-transformers** — local embeddings model (all-MiniLM-L6-v2)
- **anthropic** — Claude API for Tier 3 LLM processing
- **SQLite** — caching layer for normalizations, embeddings, LLM responses, feedback
- **PyYAML** — configuration
- **Click** — CLI framework
- **pytest** — testing

---

## Critical Design Decisions

### 1. All Monetary Values Use `Decimal`, Never `float`

This is non-negotiable. Floating point arithmetic produces rounding errors that compound across thousands of invoices. Every amount field, every calculation involving money, every threshold comparison MUST use `decimal.Decimal`. Import it everywhere.

```python
from decimal import Decimal, ROUND_HALF_UP

# CORRECT
amount = Decimal("1234.56")
threshold = Decimal("5000.00")

# WRONG — never do this
amount = 1234.56
threshold = 5000.0
```

### 2. Tiered LLM Processing (Cost Control Is Survival)

The gain-share model means unpredictable revenue. LLM costs must be controlled aggressively.

- **Tier 1 (Rule-based, zero cost):** Exact matching, statistical calculations, keyword matching, Benford's Law, z-scores, regex. This handles 60–70% of all detection logic.
- **Tier 2 (Local embeddings, minimal cost):** sentence-transformers model running locally. Used for text similarity, item classification, duplicate description matching. Cache embeddings in SQLite.
- **Tier 3 (Claude API, expensive):** Only for ambiguous cases that Tier 1 and 2 cannot resolve. Always check cache before calling. Batch multiple items into single prompts (up to 10 per call). Track costs per customer. Monthly budget cap: $200/customer.

**The rule: Never call the LLM if a rule or embedding can answer the question.**

### 3. SQLite Cache Is the Memory System

The `CacheManager` (src/normalization/cache.py) is the central persistence layer. It stores:
- Vendor name → canonical ID mappings (so normalization improves over time)
- Line item description → category classifications (so LLM calls drop 80%+ after first run)
- Text → embedding vectors (so embeddings are computed once per unique text)
- Prompt → LLM response (so identical questions never hit the API twice)
- User feedback on findings (so false positive suppression improves)
- Vendor behavioral baselines (so anomaly detection has history)

**Every function that could potentially call the LLM must check cache first.**

### 4. Every Detection Module Inherits from BaseDetector

All 8 modules follow the same interface:

```python
class SomeDetector(BaseDetector):
    def get_module_name(self) -> ModuleName: ...
    def get_required_fields(self) -> list[str]: ...
    def get_optional_fields(self) -> list[str]: ...
    def detect(self, invoice_data, vendor_data=None, supplementary_data=None) -> list[Finding]: ...
```

This ensures consistency and lets the ModuleRunner orchestrate all modules uniformly. Modules must gracefully handle missing optional data (run in degraded mode, not crash).

### 5. No Magic Numbers in Code

Every configurable threshold lives in `config/thresholds.yaml`. The code reads from config, never hardcodes values. If you need a new threshold, add it to the YAML file and reference it from there.

```python
# CORRECT
threshold = self.config["duplicate_detection"]["fuzzy_composite_threshold"]

# WRONG
threshold = 0.85
```

### 6. False Positives Are the Enemy

A customer who sees 50 flags where 35 are garbage loses trust permanently. The system is designed around precision-first:

- Start with high confidence thresholds (only flag what you're almost certain about)
- Cross-module correlation amplifies weak signals (single module flag is weak; 3 modules flagging same vendor is strong)
- Conflict resolution suppresses contradictory signals before they reach the customer
- Feedback loop learns from customer verdicts and suppresses recurring false positives
- Tiered alerting ensures CFOs only see 3–5 critical alerts, not 50 mixed-quality flags

### 7. Modules Can Conflict — The Adjudication Engine Resolves

The ConflictResolver handles known contradiction patterns:
- Duplicate detection vs. Contract compliance: if both invoices match separate contract milestones, downgrade duplicate confidence
- Price creep vs. Market price: if market prices rose more than the vendor's prices, suppress the creep flag
- Split invoicing vs. Vendor behavior: if weekly invoicing is the vendor's normal pattern, suppress the split flag
- Multi-module corroboration: if 3+ modules flag the same vendor for different issues, ESCALATE all findings

---

## The 8 Detection Modules — Quick Reference

| # | Module | What It Detects | Key Algorithm | Min Data Needed |
|---|--------|----------------|---------------|-----------------|
| 1 | Duplicate Invoices | Same transaction billed twice (exact, near-exact, reformatted, cross-PO) | Fuzzy matching + embeddings + LLM for ambiguous cases | invoice_number, vendor, date, amount |
| 2 | Price Creep | Gradual vendor price increases that exceed contracts or norms | Time-series regression, cumulative change, single-jump detection | vendor, item description, unit_price, date |
| 3 | Phantom Services | Charges for services never rendered or goods never delivered | PO gap analysis, vagueness scoring, category mismatch, receipt verification | vendor, line items, amount |
| 4 | Vendor Collusion | Related vendors coordinating to overbill | Network graph (shared attributes), timing correlation, Benford's Law | vendor master data (address, phone, tax_id, bank) |
| 5 | Contract Compliance | Billing that violates agreed contract terms | Rate comparison, payment term parsing, scope boundary checking | contract terms JSON (required) |
| 6 | Vendor Behavior | Sudden changes in vendor invoicing patterns (possible BEC/fraud) | Baseline profiling + z-score anomaly detection + drift analysis | vendor, amount, date (6+ months history) |
| 7 | Market Price | Paying above market rate for goods/services | Benchmark construction, percentile ranking, vendor premium index | vendor, item description, unit_price (3+ vendors per category) |
| 8 | Split Invoicing | Large purchases split into smaller invoices to bypass approval thresholds | Threshold clustering, temporal aggregation, Benford's Law, PO linkage | vendor, amount, date, approval thresholds |

---

## Data Model — Key Types

Defined in `src/utils/constants.py`:

- **InvoiceRecord**: One invoice with header fields + list of LineItems
- **LineItem**: One line from an invoice with description, category, quantity, price
- **VendorRecord**: Canonical vendor entity with all known attributes
- **ContractTerms**: Structured contract data (rates, payment terms, scope, dates)
- **Finding**: Output of a detection module — the core output unit of the system
- **DataQualityReport**: Assessment of input data completeness and module readiness

The **Finding** dataclass is the most important. Every detection result becomes a Finding with: module source, severity, confidence score, vendor/invoice references, dollar amount at risk, human-readable description, evidence dict, recommended action, and cross-module correlation links.

---

## Normalization Pipeline — The Foundation

**This must work correctly for anything else to work.** The pipeline runs in order:

1. **VendorNormalizer**: Resolves "ABC Corp" / "ABC Corporation" / "A.B.C. Corp" into one canonical entity. Uses rapidfuzz token_sort_ratio. Auto-merges at >90% similarity, flags for review at 75–90%, creates new entity below 75%.

2. **ItemNormalizer**: Maps "Consulting - March" and "Mar consulting svc" to the same taxonomy category. Uses 3-tier classification: keywords → embeddings → LLM. Results cached permanently.

3. **AmountNormalizer**: Standardizes currency formats, separates tax, handles credits/negatives, validates line item sums against invoice totals.

---

## Testing Strategy

- **Unit tests**: Every normalization function, every detection method, every utility function
- **Integration tests**: Full pipeline on generated sample data, validated against fraud_key.json
- **Validation targets**:
  - Duplicate detection recall: >80%
  - Price creep recall: >75%
  - Phantom services recall: >70%
  - Vendor collusion recall: >65%
  - Contract compliance recall: >80%
  - Vendor behavior recall: >75%
  - Split invoicing recall: >70%
  - Overall false positive rate: <20%

The sample data generator (`scripts/generate_sample_data.py`) creates 5,000 invoices across 150 vendors with 56 injected fraud items across all 8 types. The `fraud_key.json` file contains ground truth for automated validation.

**Always use `--seed 42` for reproducible test data.**

**Mock the LLM client in tests — never make real API calls during testing.**

---

## Performance Targets

| Dataset Size | Max Processing Time (excl. LLM) | Max LLM Cost |
|---|---|---|
| 5,000 invoices | 5 minutes | $5 |
| 10,000 invoices | 10 minutes | $8 |
| 50,000 invoices | 30 minutes | $15 |

- Embedding cache hit rate: >80% on second run for same customer
- LLM cache hit rate: >80% on second run
- Peak memory: <2GB for datasets up to 50,000 invoices

**Key optimizations:**
- Duplicate detection uses blocking (same vendor + date window) to avoid O(n²)
- Embeddings are batch-computed and cached in SQLite
- LLM calls are batched (10 items per prompt for classification)
- Vendor statistics are pre-computed once, shared across modules
- Large datasets processed in chunks (10,000 rows)

---

## CLI Commands

```bash
# Generate test data
silentauditor generate-sample-data --output-dir data/ --seed 42

# Check data quality before analysis
silentauditor quality-check -i invoices.csv

# Run full analysis
silentauditor analyze \
  -i invoices.csv \
  -v vendors.csv \
  -c contracts.json \
  -a approval_thresholds.json \
  -r goods_receipts.csv \
  --erp netsuite \
  --output-dir ./results \
  --customer-id acme_corp \
  --verbose

# Validate accuracy against known fraud
silentauditor validate \
  --input-dir data/ \
  --fraud-key data/sample_invoices/fraud_key.json

# Record feedback on a finding
silentauditor feedback \
  --finding-id SA-abc123 \
  --verdict false_positive \
  --customer-id acme_corp \
  --notes "This is a recurring subscription"
```

---

## Code Conventions

### Style
- Type hints on ALL function parameters and return types
- Docstrings on ALL classes and public methods
- Use `logging.getLogger(__name__)` in every module (never `print()`)
- Imports ordered: stdlib → third-party → local, separated by blank lines
- f-strings for string formatting (not `.format()` or `%`)

### Logging Levels
- **INFO**: Major pipeline steps, summary stats, module start/complete
- **DEBUG**: Detailed processing info, individual invoice analysis, cache hits/misses
- **WARNING**: Degraded mode activation, missing optional data, approaching cost limits
- **ERROR**: Module failures, file I/O errors, API failures (after retries exhausted)

### Error Handling
- Every file I/O operation wrapped in try/except
- Every LLM API call has retry with exponential backoff (3 attempts: 2s, 4s, 8s)
- No bare `except:` clauses — always catch specific exceptions
- Module failures are caught by ModuleRunner — one module crashing does not stop others
- Missing API key: system continues with Tier 1 and Tier 2 only (log warning, don't crash)

### File Organization
- One class per file for major components
- Utility functions grouped by domain (statistics.py, similarity.py, date_utils.py)
- Constants and enums in constants.py
- All test files mirror source structure: `src/detection/price_creep.py` → `tests/test_price_creep.py`

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | No (Tier 1-2 work without it) | Claude API key for Tier 3 LLM processing |
| `SILENTAUDITOR_DB_PATH` | No (defaults to `~/.silentauditor/cache.db`) | Custom SQLite cache location |
| `SILENTAUDITOR_LOG_LEVEL` | No (defaults to INFO) | Logging level override |

---

## Common Tasks

### Adding a new detection module
1. Create `src/detection/new_module.py` inheriting from `BaseDetector`
2. Implement all abstract methods: `get_module_name()`, `get_required_fields()`, `get_optional_fields()`, `detect()`
3. Add module name to `ModuleName` enum in constants.py
4. Add default thresholds to `config/thresholds.yaml`
5. Register in `main.py` analyze command
6. Add module weight to `RiskScorer.MODULE_WEIGHTS`
7. Add conflict resolution rules to `ConflictResolver` if applicable
8. Add fraud injection for this type to `generate_sample_data.py`
9. Write tests and update integration test

### Adjusting detection sensitivity
Edit `config/thresholds.yaml`. Key levers:
- Raise confidence thresholds → fewer flags, higher precision, more missed fraud
- Lower confidence thresholds → more flags, lower precision, less missed fraud
- Adjust amount minimums → ignore small-dollar findings
- Adjust date windows → wider windows catch more but increase false positives

### Adding a new ERP template
Create `config/erp_templates/{erp_name}.yaml` with column name mappings:
```yaml
mappings:
  invoice_number: "Document Number"
  vendor_name: "Vendor Name"
  invoice_date: "Posting Date"
  total_amount: "Amount in Doc. Currency"
  line_item_description: "Item Text"
  unit_price: "Net Price"
  quantity: "Order Quantity"
  po_number: "Purchase Order"
  payment_date: "Clearing Date"
```

### Debugging false positives
1. Run with `--verbose` to see detailed module output
2. Check which module produced the finding (Finding.module field)
3. Look at the evidence dict for scoring details
4. Use `silentauditor feedback` to mark as false positive
5. After 3 FP marks for same pattern, system auto-suppresses future occurrences
6. Check `false_positive_manager.suggest_threshold_adjustments()` for tuning recommendations

### Running without an API key
The system works without an Anthropic API key — it runs all Tier 1 (rule-based) and Tier 2 (local embedding) processing. Only Tier 3 features are skipped:
- Ambiguous duplicate pair assessment (pairs in 70–85% zone won't get LLM review)
- Complex line item classification (items that don't match keywords or embeddings default to "Uncategorized")
- Scope boundary assessment for contract compliance (falls back to embedding similarity only)

Detection accuracy drops approximately 10–15% without Tier 3, but the system is still functional.

---

## Domain Knowledge — AP Fraud Patterns

Understanding WHY each fraud type exists helps write better detection logic:

**Duplicate invoices** happen because vendors accidentally resend, ERP systems import twice, or fraudsters intentionally rebill. The sophisticated ones change the invoice number or tweak amounts by a few dollars.

**Price creep** happens because vendors test boundaries — 2% here, 3% there — knowing mid-market companies rarely track unit prices over time. It's often not malicious, but it's always recoverable money.

**Phantom services** happen when an insider creates fake invoices, or when a vendor bills for work not completed. Vague descriptions ("consulting services" with no detail) are the biggest red flag.

**Vendor collusion** happens when related entities (same owner, shared office) pretend to be separate vendors to inflate prices or rig bids. Shared addresses and bank accounts are the giveaway.

**Contract violations** happen through drift — vendors slowly deviate from agreed terms, or AP teams don't verify every invoice against the contract. The money is in missed discounts and rate overcharges.

**Behavior anomalies** happen when a vendor's account is compromised (BEC fraud — bank account change is the #1 indicator) or when internal fraud changes invoicing patterns.

**Split invoicing** happens when someone breaks a large purchase into smaller invoices to stay below approval thresholds that would require CFO or board review.

---

## What Makes This Project Different From Generic Fraud Detection

1. **Mid-market focused**: Enterprise fraud systems cost $100K+/year. This targets companies spending $5M–$100M with $30K–$75K ACV.

2. **AP-specific**: Not payment fraud, not credit card fraud — accounts payable invoice fraud, which has different patterns and data structures.

3. **Gain-share model**: Revenue depends on FINDING REAL FRAUD. False positives = wasted customer time. Missed fraud = lost revenue. Precision matters more than recall.

4. **Cross-client intelligence**: As the customer base grows, vendor fraud patterns learned from Customer A immediately help detect fraud at Customer B. This is the data moat.

5. **Continuous monitoring, not one-time audit**: Unlike quarterly forensic audits, this runs on every new invoice. The vendor behavior baseline gets stronger over time.

---

## Reminders When Working On This Codebase

- **Test with messy data, not clean data.** Real invoices have OCR errors, inconsistent naming, missing fields, mixed formats. The sample data generator includes a messy variant for this reason.
- **Check cache before computing anything expensive.** If you're writing code that calls the LLM or computes embeddings, verify it checks CacheManager first.
- **Every finding needs an amount_at_risk.** CFOs think in dollars, not anomaly scores. A finding without a dollar estimate is useless for the gain-share model.
- **Don't crash the pipeline.** If one module fails, catch the exception, log it, and continue with other modules. The customer should still get partial results.
- **Decimal, not float.** Always. For money. Every time. No exceptions.