"""SilentAuditor CLI — AI-powered accounts payable fraud detection.

Entry point for the full analysis pipeline, data quality checks,
sample-data generation, accuracy validation, and feedback recording.
"""

import json
import logging
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

import click
import pandas as pd
import yaml

from config import settings

logger = logging.getLogger("silentauditor")


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(custom_path: str = None) -> dict:
    path = Path(custom_path) if custom_path else settings.DEFAULT_CONFIG_PATH
    if not path.exists():
        click.echo(f"Config not found: {path}", err=True)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_file(path: str, label: str) -> pd.DataFrame:
    from src.ingestion.file_loader import FileLoader
    p = Path(path)
    if not p.exists():
        click.echo(f"Error: {label} file not found: {path}", err=True)
        sys.exit(1)
    try:
        return FileLoader().load(path)
    except ValueError as exc:
        click.echo(f"Error loading {label}: {exc}", err=True)
        click.echo("Supported formats: .csv, .xlsx, .xls, .json, .tsv")
        sys.exit(1)


def _numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["total_amount", "unit_price", "quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "invoice_id" not in df.columns and "invoice_number" in df.columns:
        df["invoice_id"] = df["invoice_number"]
    return df


# ------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------

@click.group()
def cli() -> None:
    """SilentAuditor — AI-Powered Accounts Payable Audit System."""
    pass


# ==================================================================
# analyze
# ==================================================================

@cli.command()
@click.option("--input", "-i", "input_path", required=True,
              help="Path to invoice CSV/Excel file")
@click.option("--vendor-master", "-v", "vendor_path",
              help="Path to vendor master CSV")
@click.option("--contracts", "-c", "contracts_path",
              help="Path to contracts JSON file")
@click.option("--approvals", "-a", "approvals_path",
              help="Path to approval thresholds JSON")
@click.option("--receipts", "-r", "receipts_path",
              help="Path to goods receipts CSV")
@click.option("--erp", help="ERP template name")
@click.option("--output-dir", "-o", default="./results",
              help="Output directory for reports")
@click.option("--config", "config_path", help="Path to custom thresholds.yaml")
@click.option("--customer-id", default="default",
              help="Customer identifier for caching")
@click.option("--skip-modules", multiple=True, help="Module names to skip")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def analyze(input_path, vendor_path, contracts_path, approvals_path,
            receipts_path, erp, output_dir, config_path, customer_id,
            skip_modules, verbose):
    """Run full SilentAuditor analysis pipeline."""
    t0 = time.perf_counter()
    _setup_logging(verbose)

    # 1-2. Config
    config = _load_config(config_path)
    click.echo("SilentAuditor — starting analysis")

    # 3. Infrastructure
    from src.normalization.cache import CacheManager
    cache = CacheManager()

    llm_client = _build_llm_client(cache, config)

    # 4. Load files
    inv_df = _load_file(input_path, "invoice")
    vendor_df = _load_file(vendor_path, "vendor master") if vendor_path else None
    receipts_df = _load_file(receipts_path, "receipts") if receipts_path else None

    contracts = None
    if contracts_path:
        try:
            from src.ingestion.file_loader import FileLoader
            contracts = FileLoader().load_contracts(contracts_path)
            logger.info("Loaded %d contracts from %s", len(contracts) if contracts else 0, contracts_path)
        except Exception as exc:
            logger.warning("Could not load contracts: %s", exc, exc_info=True)
            click.echo(f"Warning: could not load contracts: {exc}", err=True)

    thresholds = None
    if approvals_path:
        try:
            from src.ingestion.file_loader import FileLoader
            thresholds = FileLoader().load_approval_thresholds(approvals_path)
        except Exception as exc:
            click.echo(f"Warning: could not load approvals: {exc}", err=True)

    # 5. Field mapping
    from src.ingestion.field_mapper import FieldMapper
    mapper = FieldMapper(cache=cache)
    if erp:
        mapping = mapper.apply_erp_template(inv_df, erp)
    else:
        mapping = mapper.auto_map(inv_df, customer_id=customer_id)

    required = {"invoice_number", "vendor_name", "invoice_date", "total_amount"}
    missing = required - set(mapping.keys())
    if missing:
        click.echo(f"Error: required fields not mapped: {missing}", err=True)
        click.echo("Available columns: " + ", ".join(inv_df.columns))
        sys.exit(1)

    inv_df = mapper.apply_mapping(inv_df, mapping)
    inv_df = _numeric_cols(inv_df)

    # 6. Data quality
    from src.ingestion.data_quality import DataQualityAssessor
    dq = DataQualityAssessor(config).assess(inv_df, vendor_df, contracts,
                                             receipts_df, thresholds)
    click.echo(f"Data quality: {dq.overall_readiness_score:.0%} readiness, "
               f"{dq.total_invoices} invoices, {dq.unique_vendors} vendors")
    if dq.overall_readiness_score < 0.3:
        click.echo("Warning: low data quality — results may be incomplete",
                    err=True)

    # 7. Normalization
    from src.normalization.vendor_normalizer import VendorNormalizer
    from src.normalization.amount_normalizer import AmountNormalizer
    VendorNormalizer(cache).normalize_all(inv_df["vendor_name"].unique().tolist())
    inv_df = AmountNormalizer().normalize_amounts(inv_df)

    # 8-9. Detection modules
    from src.detection.base_detector import ModuleRunner
    from src.detection.duplicate_invoices import DuplicateInvoiceDetector
    from src.detection.price_creep import PriceCreepDetector
    from src.detection.phantom_services import PhantomServicesDetector
    from src.detection.vendor_collusion import VendorCollusionDetector
    from src.detection.contract_compliance import ContractComplianceDetector
    from src.detection.vendor_behavior import VendorBehaviorDetector
    from src.detection.market_price import MarketPriceDetector
    from src.detection.split_invoicing import SplitInvoiceDetector

    module_classes = [
        DuplicateInvoiceDetector, PriceCreepDetector,
        PhantomServicesDetector, VendorCollusionDetector,
        ContractComplianceDetector, VendorBehaviorDetector,
        MarketPriceDetector, SplitInvoiceDetector,
    ]

    runner = ModuleRunner(config, cache, llm_client)
    skip_set = {s.lower() for s in skip_modules}
    for cls in module_classes:
        inst = cls(config, cache, llm_client)
        if inst.get_module_name().value in skip_set:
            click.echo(f"  Skipping {inst.get_module_name().value}")
            continue
        runner.register_module(cls)

    supp = {}
    if contracts:
        supp["contracts"] = contracts
        logger.info("Supplementary data: %d contracts added", len(contracts))
    if thresholds:
        supp["approval_thresholds"] = thresholds
        logger.info("Supplementary data: %d approval thresholds added", len(thresholds))
    if receipts_df is not None:
        supp["receipts"] = receipts_df
        logger.info("Supplementary data: %d receipts added", len(receipts_df))

    findings = runner.run_all(inv_df, vendor_data=vendor_df,
                              supplementary_data=supp if supp else None)

    # 10-11. Adjudication
    from src.adjudication.conflict_resolver import ConflictResolver
    from src.adjudication.false_positive_manager import FalsePositiveManager
    from src.adjudication.risk_scorer import RiskScorer

    findings = ConflictResolver(config).resolve(findings)
    findings = FalsePositiveManager(cache).apply_suppression_rules(
        findings, customer_id)

    # 12. Scoring
    scorer = RiskScorer()
    vendor_scores = scorer.score_vendors(findings)
    invoice_scores = scorer.score_invoices(findings)

    # 13. Alerts
    from src.reporting.alert_manager import AlertManager
    alerts = AlertManager(config).categorize_alerts(findings)
    alert_summary = AlertManager(config).generate_alert_summary(alerts)

    # 14. Reports
    from src.reporting.report_generator import ReportGenerator
    from src.reporting.dashboard_data import DashboardDataGenerator
    ReportGenerator(output_dir).generate_all_reports(
        findings, vendor_scores, invoice_scores, alert_summary, dq)
    DashboardDataGenerator().generate_dashboard_json(
        findings, vendor_scores, invoice_scores, alert_summary, dq,
        output_path=os.path.join(output_dir, "dashboard_data.json"))

    # 15. Console summary
    elapsed = time.perf_counter() - t0
    active = [f for f in findings if not f.suppressed]
    crit = [f for f in active if f.severity.value == "critical"]
    crit_amount = sum(float(f.amount_at_risk) for f in crit)

    click.echo("")
    click.echo("=" * 60)
    click.echo("  Analysis Complete")
    click.echo("=" * 60)
    click.echo(f"  Findings:   {len(active)} active, "
               f"{len(findings) - len(active)} suppressed")
    click.echo(f"  Critical:   {len(crit)} (${crit_amount:,.2f} at risk)")
    click.echo(f"  Time:       {elapsed:.1f}s")
    click.echo(f"  Reports:    {output_dir}/")
    click.echo("=" * 60)

    cache.close()


# ==================================================================
# quality-check
# ==================================================================

@cli.command("quality-check")
@click.option("--input", "-i", "input_path", required=True,
              help="Path to invoice file")
@click.option("--verbose", is_flag=True)
def quality_check(input_path: str, verbose: bool) -> None:
    """Run data quality scorecard only."""
    _setup_logging(verbose)
    inv_df = _load_file(input_path, "invoice")

    from src.ingestion.field_mapper import FieldMapper
    from src.ingestion.data_quality import DataQualityAssessor
    from src.normalization.cache import CacheManager

    cache = CacheManager()
    mapper = FieldMapper(cache=cache)
    mapping = mapper.auto_map(inv_df)

    required = {"invoice_number", "vendor_name", "invoice_date", "total_amount"}
    if not required.issubset(mapping.keys()):
        inv_df = _numeric_cols(inv_df)
    else:
        inv_df = mapper.apply_mapping(inv_df, mapping)
        inv_df = _numeric_cols(inv_df)

    dq = DataQualityAssessor().assess(inv_df)
    DataQualityAssessor().print_report(dq)
    cache.close()


# ==================================================================
# generate-sample-data
# ==================================================================

@cli.command("generate-sample-data")
@click.option("--output-dir", "-o", default="./data")
@click.option("--seed", default=42, type=int)
def generate_sample_data(output_dir: str, seed: int) -> None:
    """Generate sample test data with injected fraud patterns."""
    from scripts.generate_sample_data import SampleDataGenerator
    gen = SampleDataGenerator(seed=seed)
    gen.generate(output_dir)


# ==================================================================
# validate
# ==================================================================

@cli.command()
@click.option("--input-dir", "-i", required=True,
              help="Directory containing generated sample data")
@click.option("--fraud-key", "-k", required=True,
              help="Path to fraud_key.json")
@click.option("--output-dir", "-o", default="./validation_results")
@click.option("--verbose", is_flag=True)
def validate(input_dir: str, fraud_key: str, output_dir: str, verbose: bool) -> None:
    """Validate detection accuracy against known fraud key."""
    _setup_logging(verbose)
    from src.normalization.cache import CacheManager
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

    base = Path(input_dir)
    config = _load_config()
    cache = CacheManager()
    llm = _build_llm_client(cache, config)

    inv_df = _load_file(str(base / "sample_invoices" / "invoices.csv"), "invoices")
    inv_df = _numeric_cols(inv_df)
    vendor_df = _load_file(str(base / "sample_vendor_master" / "vendors.csv"), "vendors")

    contracts_path = base / "sample_contracts" / "contracts.json"
    contracts = json.loads(contracts_path.read_text()) if contracts_path.exists() else []
    thresh_path = base / "sample_invoices" / "approval_thresholds.json"
    thresholds = json.loads(thresh_path.read_text()) if thresh_path.exists() else []
    receipts_path = base / "sample_invoices" / "goods_receipts.csv"
    receipts_df = _load_file(str(receipts_path), "receipts") if receipts_path.exists() else None

    runner = ModuleRunner(config, cache, llm)
    for cls in [DuplicateInvoiceDetector, PriceCreepDetector,
                PhantomServicesDetector, VendorCollusionDetector,
                ContractComplianceDetector, VendorBehaviorDetector,
                MarketPriceDetector, SplitInvoiceDetector]:
        runner.register_module(cls)

    supp = {"contracts": contracts, "approval_thresholds": thresholds}
    if receipts_df is not None:
        supp["receipts"] = receipts_df

    findings = runner.run_all(inv_df, vendor_data=vendor_df, supplementary_data=supp)
    findings = ConflictResolver(config).resolve(findings)
    active = [f for f in findings if not f.suppressed]

    with open(fraud_key) as f:
        key = json.load(f)

    # Per-type accuracy
    click.echo("\nValidation Results:")
    click.echo("-" * 60)
    all_types = sorted({fi["fraud_type"] for fi in key["fraud_items"]})
    total_det, total_tot = 0, 0
    for ftype in all_types:
        type_frauds = [fi for fi in key["fraud_items"] if fi["fraud_type"] == ftype]
        detected = 0
        for fraud_item in type_frauds:
            fids = set(fraud_item.get("invoice_ids", []))
            fvendor = fraud_item.get("vendor", "")
            for finding in active:
                if fids & set(finding.invoice_ids) or finding.vendor_name in fvendor:
                    detected += 1
                    break
        rate = detected / len(type_frauds) * 100 if type_frauds else 0
        click.echo(f"  {ftype:<28s}  {detected}/{len(type_frauds)}  ({rate:.0f}%)")
        total_det += detected
        total_tot += len(type_frauds)

    rate = total_det / total_tot * 100 if total_tot else 0
    click.echo("-" * 60)
    click.echo(f"  {'OVERALL':<28s}  {total_det}/{total_tot}  ({rate:.0f}%)")

    os.makedirs(output_dir, exist_ok=True)
    report = {"per_type": {}, "overall_rate": rate, "total_findings": len(active)}
    for ftype in all_types:
        type_frauds = [fi for fi in key["fraud_items"] if fi["fraud_type"] == ftype]
        det = sum(1 for fi in type_frauds
                  if any(set(fi.get("invoice_ids", [])) & set(f.invoice_ids)
                         or f.vendor_name in fi.get("vendor", "")
                         for f in active))
        report["per_type"][ftype] = {"total": len(type_frauds), "detected": det}
    with open(os.path.join(output_dir, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    click.echo(f"\nReport saved to {output_dir}/validation_report.json")
    cache.close()


# ==================================================================
# feedback
# ==================================================================

@cli.command()
@click.option("--finding-id", required=True)
@click.option("--verdict", required=True,
              type=click.Choice(["confirmed_fraud", "false_positive", "legitimate"]))
@click.option("--customer-id", default="default")
@click.option("--notes", default="")
def feedback(finding_id: str, verdict: str, customer_id: str, notes: str) -> None:
    """Record feedback on a finding."""
    from src.normalization.cache import CacheManager
    from src.adjudication.false_positive_manager import FalsePositiveManager
    from src.utils.constants import FeedbackVerdict

    cache = CacheManager()
    mgr = FalsePositiveManager(cache)
    mgr.record_feedback(finding_id, customer_id,
                        FeedbackVerdict(verdict), notes)
    click.echo(f"Recorded: {finding_id} → {verdict}")
    cache.close()


# ------------------------------------------------------------------
# LLM client builder (graceful when API key missing)
# ------------------------------------------------------------------

def _build_llm_client(cache: object, config: dict) -> object:
    """Build an LLM client; fall back to Tier 1-2 only if no API key."""
    try:
        from src.utils.llm_client import LLMClient
        client = LLMClient(cache=cache, config=config)
        return client
    except Exception as exc:
        logger.warning("LLM client init failed: %s — Tier 3 disabled", exc)

        class _FallbackLLM:
            """Stub that provides Tier 1-2 methods only."""

            def classify_by_keywords(self, desc: str, kw_map: dict) -> str | None:
                if not desc or not kw_map:
                    return None
                dl = desc.lower()
                best, bl = None, 0
                for k, v in kw_map.items():
                    if k.lower() in dl and len(k) > bl:
                        best, bl = v, len(k)
                return best

            def score_description_vagueness(self, desc: str) -> int:
                return 50

            def compute_text_similarity(self, a: str, b: str) -> float:
                return 0.5

            def assess_duplicate_pair(self, a: dict, b: dict, **kw: object) -> dict:
                return {"is_duplicate": False, "confidence": 0, "reasoning": ""}

            def classify_line_item(self, desc: str, tax: dict, cid: str) -> dict:
                return {"category": "Uncategorized", "source": "rule",
                        "confidence": 0.5, "subcategory": None, "item_type": None}

            def get_embedding(self, text: str) -> object:
                import numpy as np
                return np.zeros(384, dtype=np.float32)

            def get_session_cost(self) -> Decimal:
                return Decimal("0")

        return _FallbackLLM()


if __name__ == "__main__":
    cli()
