"""Project-wide constants, enums, and data models for SilentAuditor."""

import uuid
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(Enum):
    """Alert severity tiers for findings."""
    CRITICAL = "critical"
    REVIEW = "review"
    INFORMATIONAL = "informational"


class FeedbackVerdict(Enum):
    """User feedback on a finding."""
    CONFIRMED_FRAUD = "confirmed_fraud"
    FALSE_POSITIVE = "false_positive"
    LEGITIMATE = "legitimate"


class ModuleName(Enum):
    """Detection module identifiers."""
    DUPLICATE_DETECTION = "duplicate_detection"
    PRICE_CREEP = "price_creep"
    PHANTOM_SERVICES = "phantom_services"
    VENDOR_COLLUSION = "vendor_collusion"
    CONTRACT_COMPLIANCE = "contract_compliance"
    VENDOR_BEHAVIOR = "vendor_behavior"
    MARKET_PRICE = "market_price"
    SPLIT_INVOICING = "split_invoicing"


class DataTier(Enum):
    """Processing tier for cost-controlled LLM usage."""
    RULE_BASED = "rule_based"
    EMBEDDING = "embedding"
    LLM = "llm"


class RiskTier(Enum):
    """Risk classification for vendors and findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FieldSource(Enum):
    """How a field mapping was determined."""
    AUTO_DETECTED = "auto_detected"
    ERP_TEMPLATE = "erp_template"
    USER_CONFIRMED = "user_confirmed"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def generate_finding_id() -> str:
    """Return a UUID-based finding ID with 'SA-' prefix."""
    return f"SA-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class LineItem:
    """A single line item within an invoice."""
    line_item_id: str
    invoice_id: str
    description: str
    total_amount: Decimal
    normalized_description: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    item_type: Optional[str] = None
    quantity: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    unit_of_measure: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"LineItem(line_item_id={self.line_item_id!r}, "
            f"description={self.description!r}, "
            f"total_amount={self.total_amount})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LineItem):
            return NotImplemented
        return (
            self.line_item_id == other.line_item_id
            and self.invoice_id == other.invoice_id
        )


@dataclass
class InvoiceRecord:
    """A normalised invoice with its line items."""
    invoice_id: str
    invoice_number: str
    vendor_id: str
    vendor_name: str
    invoice_date: date
    total_amount: Decimal
    line_items: list[LineItem]
    raw_data: dict
    po_number: Optional[str] = None
    payment_date: Optional[date] = None
    payment_status: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"InvoiceRecord(invoice_id={self.invoice_id!r}, "
            f"invoice_number={self.invoice_number!r}, "
            f"vendor_name={self.vendor_name!r}, "
            f"total_amount={self.total_amount})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InvoiceRecord):
            return NotImplemented
        return self.invoice_id == other.invoice_id


@dataclass
class VendorRecord:
    """Canonical vendor information with aliases."""
    vendor_id: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    address: Optional[str] = None
    phone: Optional[str] = None
    tax_id: Optional[str] = None
    bank_account_last4: Optional[str] = None
    contact_person: Optional[str] = None
    category: Optional[str] = None
    date_added: Optional[date] = None

    def __repr__(self) -> str:
        return (
            f"VendorRecord(vendor_id={self.vendor_id!r}, "
            f"canonical_name={self.canonical_name!r}, "
            f"aliases={self.aliases})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VendorRecord):
            return NotImplemented
        return self.vendor_id == other.vendor_id


@dataclass
class ContractRate:
    """A contracted rate for a specific item or service."""
    item_description: str
    unit: str
    rate: Decimal

    def __repr__(self) -> str:
        return (
            f"ContractRate(item_description={self.item_description!r}, "
            f"unit={self.unit!r}, rate={self.rate})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContractRate):
            return NotImplemented
        return (
            self.item_description == other.item_description
            and self.unit == other.unit
            and self.rate == other.rate
        )


@dataclass
class VolumeDiscount:
    """A volume-based discount tier in a contract."""
    threshold_quantity: int
    discount_pct: Decimal

    def __repr__(self) -> str:
        return (
            f"VolumeDiscount(threshold_quantity={self.threshold_quantity}, "
            f"discount_pct={self.discount_pct})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VolumeDiscount):
            return NotImplemented
        return (
            self.threshold_quantity == other.threshold_quantity
            and self.discount_pct == other.discount_pct
        )


@dataclass
class ContractTerms:
    """Terms and conditions from a vendor contract."""
    vendor_id: str
    contract_start_date: date
    contract_end_date: date
    auto_renewal: bool
    auto_renewal_notice_days: int
    payment_terms: str
    rates: list[ContractRate]
    volume_discounts: list[VolumeDiscount]
    scope_of_work: list[str]
    annual_escalation_pct: Optional[Decimal] = None

    def __repr__(self) -> str:
        return (
            f"ContractTerms(vendor_id={self.vendor_id!r}, "
            f"start={self.contract_start_date}, "
            f"end={self.contract_end_date})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContractTerms):
            return NotImplemented
        return (
            self.vendor_id == other.vendor_id
            and self.contract_start_date == other.contract_start_date
            and self.contract_end_date == other.contract_end_date
        )


@dataclass
class Finding:
    """A detection finding produced by any detection module."""
    module: ModuleName
    severity: Severity
    confidence: float
    vendor_id: str
    vendor_name: str
    invoice_ids: list[str]
    amount_at_risk: Decimal
    description: str
    evidence: dict
    recommended_action: str
    finding_id: str = field(default_factory=generate_finding_id)
    module_correlations: list[str] = field(default_factory=list)
    corroborating_modules: list[str] = field(default_factory=list)
    corroboration_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    suppressed: bool = False
    suppression_reason: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"Finding(finding_id={self.finding_id!r}, "
            f"module={self.module.value}, "
            f"severity={self.severity.value}, "
            f"confidence={self.confidence:.2f}, "
            f"amount_at_risk={self.amount_at_risk})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Finding):
            return NotImplemented
        return self.finding_id == other.finding_id


@dataclass
class ApprovalThreshold:
    """An approval authority level with optional amount cap."""
    level: str
    max_amount: Optional[Decimal] = None

    def __repr__(self) -> str:
        return (
            f"ApprovalThreshold(level={self.level!r}, "
            f"max_amount={self.max_amount})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ApprovalThreshold):
            return NotImplemented
        return self.level == other.level and self.max_amount == other.max_amount


@dataclass
class DataQualityReport:
    """Summary of data quality assessment after ingestion."""
    total_invoices: int
    date_range: dict
    unique_vendors: int
    fields_present: dict
    module_readiness: dict
    data_quality_issues: list[str]
    overall_readiness_score: float

    def __repr__(self) -> str:
        return (
            f"DataQualityReport(total_invoices={self.total_invoices}, "
            f"unique_vendors={self.unique_vendors}, "
            f"readiness={self.overall_readiness_score:.2f})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataQualityReport):
            return NotImplemented
        return (
            self.total_invoices == other.total_invoices
            and self.unique_vendors == other.unique_vendors
            and self.overall_readiness_score == other.overall_readiness_score
        )
