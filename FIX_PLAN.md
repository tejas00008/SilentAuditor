# SilentAuditor Precision Fix Plan

**Blind test results:** 100% recall, 2.5% precision (677 TP in 27,011 findings)
**Target:** 100% recall, 25-35% precision (~650 TP in 2,500-4,000 findings)

---

## Fix 1: Phantom Services Intra-Module Deduplication

**Impact: Cuts ~14,000 false positives. Single biggest win.**

The five detection methods run independently. The same invoice line triggers 3-4 separate findings (no PO + vague description + category mismatch + one-off charge). 18,678 total findings at 1.7% precision.

| File | Function | Change |
|------|----------|--------|
| `phantom_services.py` | `detect()` | After all 5 methods run, add dedup pass: group findings by invoice_id, keep highest-confidence finding per invoice. Merge evidence dicts from suppressed findings into survivor. |
| `phantom_services.py` | `_detect_po_gaps()` | Raise min amount from $500 to $1,000. Add vendor history check: if vendor has 10+ invoices without POs, suppress (PO-less billing is their norm). |
| `phantom_services.py` | `_detect_vagueness()` | Raise vagueness threshold from 40 to 25. Skip invoices with receipt match. Add $2,000 amount floor. |
| `phantom_services.py` | `_detect_category_mismatch()` | Require < 1% of history AND amount > $3,000 (currently 2% and $1,000). Skip vendors with < 5 total invoices. |
| `phantom_services.py` | `_detect_receipt_gaps()` | Calculate dataset receipt coverage. If < 50% have receipts, disable method entirely with warning. Add service-type exclusion (consulting, legal, staffing don't need receipts). |
| `phantom_services.py` | `detect()` | Add contract scope awareness: if contracts exist and line item matches scope, suppress category_mismatch and one_off findings. |

**Expected: 18,678 → 2,000-3,000 findings. Precision 1.7% → 12-15%.**

---

## Fix 2: Cross-Module Finding Deduplication in ModuleRunner

**Impact: Cuts ~4,000 duplicate findings across all modules.**

ModuleRunner.run_all() blindly extends all findings. Same invoice flagged by phantom_services, market_price, and split_invoicing = 3 separate findings.

| File | Function | Change |
|------|----------|--------|
| `base_detector.py` | `ModuleRunner.run_all()` | After collecting all findings, group by (invoice_id, vendor). For groups with 2+ findings, keep highest-confidence, annotate with corroborating_modules list. |
| `base_detector.py` | `ModuleRunner.run_all()` | Add minimum confidence gate: discard findings with confidence < 0.50. |
| `constants.py` | `Finding` dataclass | Add fields: `corroborating_modules: list[str]` and `corroboration_count: int`. |

**Expected: Total findings drop 30-40% from cross-module overlap elimination.**

---

## Fix 3: Fix Conflict Resolver Rule 5 (Corroboration Escalation)

**Impact: Stops false positives from being promoted to CRITICAL.**

Rule 5 escalates ALL findings when 2+ modules flag the same vendor. Since phantom_services and market_price flag nearly every vendor, Rule 5 promotes almost everything to CRITICAL.

| File | Function | Change |
|------|----------|--------|
| `conflict_resolver.py` | `_rule_multi_module()` | Raise corroboration threshold from 2 to 3 distinct modules. |
| `conflict_resolver.py` | `_rule_multi_module()` | Add quality gate: only count module toward corroboration if its best finding for that vendor has confidence >= 0.75. |
| `conflict_resolver.py` | `_rule_multi_module()` | Instead of escalating severity, set a `corroborated` flag. Let alert_manager decide handling. |

**Expected: CRITICAL findings drop from 26,923 → 500-1,000.**

---

## Fix 4: Market Price Threshold Tightening

**Impact: Cuts ~5,000 false positives.**

Flagging at p75 + 15% is too aggressive (25% of prices are naturally above p75). Module creates 3 separate findings per vendor with no dedup.

| File | Function | Change |
|------|----------|--------|
| `market_price.py` | `_score_deviations()` | Change threshold from p75 + 15% to p90 + 10%. |
| `market_price.py` | `_score_deviations()` | Add min absolute deviation: flag only if per-unit difference > $50 OR total overcharge > $500. |
| `market_price.py` | `_vendor_premium()` | If vendor already has item-level findings, merge premium into evidence instead of new finding. |
| `market_price.py` | `_renegotiation()` | Raise savings threshold from $1,000 to $5,000. |
| `thresholds.yaml` | `market_price` | Replace `deviation_flag_above_p75_pct: 15` with `deviation_flag_above_p90_pct: 10`. Add `min_absolute_deviation_dollars: 50`, `min_total_overcharge: 500`. |

**Expected: 6,537 → 800-1,200 findings. Precision 2.8% → 15-20%.**

---

## Fix 5: Split Invoicing False Positive Suppression

**Impact: Cuts ~1,200 false positives.**

Overlapping temporal windows create duplicates. PO-linked splits fire CRITICAL on normal multi-invoice projects. No construction awareness.

| File | Function | Change |
|------|----------|--------|
| `split_invoicing.py` | `_temporal_aggregation()` | Deduplicate across window sizes: if 7-day window finds cluster, skip 14/30-day for those invoices. |
| `split_invoicing.py` | `_po_project_splits()` | Downgrade CRITICAL to REVIEW. Suppress if descriptions contain progress/draw/retainage/milestone/phase. |
| `split_invoicing.py` | `_threshold_clustering()` | Raise min cluster size from 3 to 4. Tighten chi-square to p < 0.01. |
| `split_invoicing.py` | `_pattern_change()` | Widen normal band: amount drop threshold 30% → 40%, spend maintenance 0.9 → 0.8. |
| `thresholds.yaml` | `split_invoicing` | Add `construction_keywords_suppress` list. Set `min_invoices_in_cluster: 4`. |

**Expected: 1,552 → 300-400 findings. Precision 9.3% → 30-40%.**

---

## Fix 6: Contract Compliance Path Fix

**Impact: Enables a module that currently produces 0 findings despite data being available.**

Executive summary says "No contract data provided" even though contracts.json was passed.

| File | Function | Change |
|------|----------|--------|
| `main.py` | `analyze` command | Add logging when building supplementary_data. Verify key name matches what ContractComplianceDetector expects. |
| `contract_compliance.py` | `detect()` | Add explicit entry log with contract count or "no data" message. |
| `data_quality.py` | `assess()` | Fix readiness check to use same supplementary_data key as module. |

**Expected: Module runs and catches the 4 rate overcharges in blind test data.**

---

## Implementation Order

| # | Fix | Files | Impact |
|---|-----|-------|--------|
| 1 | Phantom services dedup | phantom_services.py | ~14,000 FP cut |
| 2 | Cross-module dedup | base_detector.py, constants.py | ~4,000 FP cut |
| 3 | Conflict resolver Rule 5 | conflict_resolver.py | ~20,000 severity fix |
| 4 | Market price thresholds | market_price.py, thresholds.yaml | ~5,000 FP cut |
| 5 | Split invoicing cleanup | split_invoicing.py, thresholds.yaml | ~1,200 FP cut |
| 6 | Contract compliance path | main.py, contract_compliance.py | Enables module |

---

## Projected Outcome

**Before:** 27,011 findings, 677 TP, 2.5% precision, 100% recall.

**After:** 2,500-4,000 findings, ~650+ TP, 20-30% precision, 98-100% recall.

The small recall risk (if any) comes from raising confidence floors. The gain-share model needs precision above 20% to be viable. These fixes get there without sacrificing novel fraud detection.
