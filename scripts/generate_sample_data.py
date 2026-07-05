#!/usr/bin/env python3
"""Generate realistic invoice data with injected fraud patterns.

Usage::

    python scripts/generate_sample_data.py --output-dir data/ --seed 42

Produces six files under *output-dir*:

* ``sample_invoices/invoices.csv``
* ``sample_invoices/goods_receipts.csv``
* ``sample_invoices/approval_thresholds.json``
* ``sample_invoices/fraud_key.json``
* ``sample_vendor_master/vendors.csv``
* ``sample_contracts/contracts.json``
"""

import argparse
import csv
import json
import logging
import sys
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TWO = Decimal("0.01")

DATE_START = date(2024, 1, 1)
DATE_END = date(2025, 6, 30)
DATE_RANGE_DAYS = (DATE_END - DATE_START).days

APPROVAL_THRESHOLDS = [
    {"level": "AP Manager", "max_amount": 5000},
    {"level": "Controller", "max_amount": 25000},
    {"level": "CFO", "max_amount": 100000},
    {"level": "Board", "max_amount": None},
]

# ---------------------------------------------------------------------------
# Vendor catalogue
# ---------------------------------------------------------------------------

_CONSTRUCTION = [
    ("Granite State Concrete", "Concrete"),
    ("Summit Steel Fabricators", "Metals"),
    ("Pacific Lumber Co", "Lumber"),
    ("Ironside Structural", "Metals"),
    ("Bluebird Plumbing", "Plumbing"),
    ("Voltaic Electrical", "Electrical"),
    ("Precision HVAC Systems", "HVAC"),
    ("Ridgeline Roofing", "Roofing"),
    ("Cornerstone Masonry", "Concrete"),
    ("Clearview Glass & Glazing", "Glass"),
    ("TerraForm Excavation", "Excavation"),
    ("ProFrame Carpentry", "Framing"),
    ("Atlas Crane Rental", "Equipment Rental"),
    ("Pinnacle Painting", "Painting"),
    ("Solid Rock Foundation", "Concrete"),
    ("BrightStar Flooring", "Flooring"),
    ("Northway Paving", "Paving"),
    ("Evergreen Landscaping", "Landscaping"),
    ("Sierra Welding", "Metals"),
    ("Cascade Drywall", "Drywall"),
    ("Metro Fire Protection", "Fire Protection"),
    ("Titan Heavy Equipment", "Equipment Rental"),
    ("BlueLine Plumbing Supply", "Pipes & Fittings"),
    ("Olympic Steel Supply", "Metals"),
    ("Keystone Demolition", "Demolition"),
]

_SERVICES = [
    ("Whitfield & Associates", "Consulting"),
    ("Bridgeport Engineering Group", "Engineering"),
    ("Haynes Legal Partners", "Legal"),
    ("Clearwater IT Solutions", "Technology"),
    ("Redstone Financial Advisory", "Accounting"),
    ("Apex Project Management", "Consulting"),
    ("NorthStar Architects", "Architecture"),
    ("Pinnacle HR Solutions", "Consulting"),
    ("DataBridge Analytics", "Technology"),
    ("GreenField Environmental", "Engineering"),
    ("Harmon Safety Consulting", "Consulting"),
    ("TrueNorth Tax Services", "Accounting"),
    ("Velocity Training Group", "Training"),
    ("Insight Research Labs", "Research"),
    ("CloudNine Hosting", "Technology"),
    ("SecureShield Cyber", "Technology"),
    ("Compliance First Advisors", "Consulting"),
    ("Quantum Design Studio", "Architecture"),
    ("BrightPath Staffing", "Staffing"),
    ("Meridian Survey Co", "Engineering"),
]

_OFFICE = [
    ("OfficeMax Pro", "Office Supplies"),
    ("PrintWorks Express", "Printing"),
    ("FreshAir HVAC Services", "HVAC Maintenance"),
    ("SparkClean Janitorial", "Cleaning"),
    ("Guardian Security", "Security"),
    ("Metro Waste Solutions", "Waste Management"),
    ("PureWater Coolers", "Facilities"),
    ("TechEdge Computers", "Hardware"),
    ("PaperTrail Supplies", "Office Supplies"),
    ("SwiftShip Courier", "Courier"),
    ("AllState Insurance Brokers", "Insurance"),
    ("Vanguard Benefits Admin", "Insurance"),
    ("Corporate Catering Co", "Food Services"),
    ("EcoGreen Recycling", "Waste Management"),
    ("SignaturePrint Marketing", "Printing"),
]

_HEALTHCARE = [
    ("MedLine Direct", "Medical Supplies"),
    ("PharmaCare Distributors", "Pharmaceuticals"),
    ("PrecisionDx Equipment", "Diagnostic Equipment"),
    ("SafeHands PPE Supply", "PPE"),
    ("LabTech Solutions", "Lab Supplies"),
    ("VitalSign Monitors", "Patient Monitoring"),
    ("BioClean Sterilization", "Sterilization"),
    ("NurseStaff Temps", "Staffing"),
    ("Imaging Systems Inc", "Diagnostic Equipment"),
    ("ClearPath Lab Services", "Lab Services"),
]

_TRANSPORT = [
    ("RoadRunner Freight", "Trucking"),
    ("SkyHigh Air Cargo", "Air Freight"),
    ("Harbor Logistics", "Ocean Freight"),
    ("QuickDrop Delivery", "Local Delivery"),
    ("FleetMaster Leasing", "Fleet"),
    ("PetroPump Fuel Co", "Fuel"),
    ("RailLine Express", "Rail"),
    ("CrossCountry Trucking", "Trucking"),
]

_EXTRA_CONSTRUCTION = [
    ("Horizon Builders", "General Contracting"),
    ("Steelcore Fabrication", "Metals"),
    ("WestCoast Concrete", "Concrete"),
    ("Pioneer Framing", "Framing"),
    ("Capitol Plumbing", "Plumbing"),
    ("Zenith Electric", "Electrical"),
    ("Frostline Insulation", "Insulation"),
    ("Rampart Fencing", "Fencing"),
    ("Beacon Scaffolding", "Equipment Rental"),
    ("HighPoint Elevators", "Elevators"),
    ("Delta Waterproofing", "Waterproofing"),
    ("Lakeview Tile & Stone", "Flooring"),
    ("Trident Marine Const", "Marine Construction"),
    ("Oakridge Cabinetry", "Millwork"),
    ("Crestwood Millwork", "Millwork"),
]

_EXTRA_SERVICES = [
    ("Summit Strategy Group", "Consulting"),
    ("Eastgate Legal Counsel", "Legal"),
    ("Prism Analytics", "Technology"),
    ("FairPoint Accounting", "Accounting"),
    ("Catalyst Innovation Lab", "Research"),
    ("BlueSky Marketing", "Marketing"),
    ("Lighthouse Compliance", "Consulting"),
    ("UrbanPlan Design", "Architecture"),
    ("TopTier Recruiters", "Staffing"),
    ("Benchmark Quality Assurance", "Consulting"),
    ("NextWave Digital", "Technology"),
    ("Sterling Communications", "Marketing"),
    ("Foundation Grant Writers", "Consulting"),
    ("Vanguard Strategy", "Consulting"),
    ("Sapphire Data Systems", "Technology"),
]

_EXTRA_OFFICE = [
    ("CleanSweep Services", "Cleaning"),
    ("Greenleaf Plant Care", "Facilities"),
    ("Brightside Window Wash", "Cleaning"),
    ("Uniform World", "Uniforms"),
    ("BrewMaster Coffee", "Food Services"),
    ("SafeVault Storage", "Storage"),
    ("ProShred Document Dest", "Document Services"),
    ("AirPure Filtration", "HVAC Maintenance"),
    ("EliteCopy Print Shop", "Printing"),
    ("DeskSpace Ergonomics", "Furniture"),
]

_EXTRA_HEALTHCARE = [
    ("SterileFirst Supplies", "Medical Supplies"),
    ("CarePoint Diagnostics", "Diagnostic Equipment"),
    ("MedTrans Ambulance", "Transport"),
    ("WellStaff Agency", "Staffing"),
    ("ProLab Chemicals", "Lab Supplies"),
    ("VisionCare Optics", "Diagnostic Equipment"),
    ("OrthoParts Supply", "Medical Supplies"),
    ("BioGenesis Research", "Research"),
    ("CardioTech Monitors", "Patient Monitoring"),
    ("PharmaDirect Wholesale", "Pharmaceuticals"),
]

_EXTRA_TRANSPORT = [
    ("ExpressLane Logistics", "Trucking"),
    ("CargoMaster Intl", "Ocean Freight"),
    ("AeroSwift Air Cargo", "Air Freight"),
    ("LocalHaul Delivery", "Local Delivery"),
    ("AutoFleet Services", "Fleet"),
    ("TankFarm Petroleum", "Fuel"),
    ("InterModal Freight", "Rail"),
    ("PackRight Shipping", "Courier"),
]

# Collusion-cluster vendors (will share addresses/bank info)
_COLLUSION_A = [
    ("Alpine Mechanical Services", "HVAC"),
    ("Alpine Electrical Works", "Electrical"),
    ("Alpine General Contracting", "General Contracting"),
]

_COLLUSION_B = [
    ("Riverside Plumbing Experts", "Plumbing"),
    ("Riverside Maintenance Group", "Maintenance"),
]

_ALL_VENDORS = (
    [(n, c, "construction") for n, c in _CONSTRUCTION]
    + [(n, c, "construction") for n, c in _EXTRA_CONSTRUCTION]
    + [(n, c, "services") for n, c in _SERVICES]
    + [(n, c, "services") for n, c in _EXTRA_SERVICES]
    + [(n, c, "office") for n, c in _OFFICE]
    + [(n, c, "office") for n, c in _EXTRA_OFFICE]
    + [(n, c, "healthcare") for n, c in _HEALTHCARE]
    + [(n, c, "healthcare") for n, c in _EXTRA_HEALTHCARE]
    + [(n, c, "transport") for n, c in _TRANSPORT]
    + [(n, c, "transport") for n, c in _EXTRA_TRANSPORT]
    + [(n, c, "construction") for n, c in _COLLUSION_A]
    + [(n, c, "construction") for n, c in _COLLUSION_B]
)

# Seasonal vendors (index into _ALL_VENDORS)
_SEASONAL_NAMES = {"Evergreen Landscaping", "FreshAir HVAC Services"}

# Line-item description templates by category
_DESCRIPTIONS = {
    "Concrete": [
        "Ready-mix concrete {qty} CY, {psi} PSI",
        "Concrete pumping services",
        "Concrete finishing labor",
        "Precast panels {qty} pcs",
    ],
    "Metals": [
        "Steel rebar #{size} grade 60, {qty} tons",
        "Structural steel W{size} beam, {qty} LF",
        "Sheet metal 16ga {qty} sheets",
        "Aluminum flashing {qty} LF",
    ],
    "Lumber": [
        "Framing lumber 2x{size} SPF, {qty} BF",
        "Plywood 3/4\" {qty} sheets",
        "Engineered I-joists {qty} pcs",
    ],
    "Plumbing": [
        "Plumbing rough-in labor {qty} hours",
        "Copper pipe 3/4\" {qty} LF",
        "Fixture installation {qty} ea",
    ],
    "Electrical": [
        "Electrical wiring labor {qty} hours",
        "Panel installation 200A",
        "Conduit EMT 3/4\" {qty} LF",
        "Light fixture installation {qty} ea",
    ],
    "HVAC": [
        "HVAC unit installation {qty} tons",
        "Ductwork fabrication {qty} LF",
        "HVAC maintenance service call",
    ],
    "Consulting": [
        "Management consulting services - {month}",
        "Strategic planning workshop",
        "Process improvement assessment",
        "Advisory services {qty} hours",
    ],
    "Engineering": [
        "Structural engineering review",
        "Civil engineering design {qty} hours",
        "Site survey and analysis",
        "MEP engineering coordination",
    ],
    "Technology": [
        "IT support services - {month}",
        "Software license renewal",
        "Cloud hosting services - {month}",
        "Network infrastructure upgrade",
        "Cybersecurity assessment",
    ],
    "Accounting": [
        "Monthly bookkeeping - {month}",
        "Tax preparation services",
        "Quarterly audit review",
        "Payroll processing - {month}",
    ],
    "Legal": [
        "Legal retainer - {month}",
        "Contract review and negotiation",
        "Compliance advisory {qty} hours",
    ],
    "Office Supplies": [
        "General office supplies - {month}",
        "Printer paper 8.5x11 {qty} cases",
        "Toner cartridges {qty} ea",
    ],
    "Cleaning": [
        "Janitorial services - {month}",
        "Floor waxing and polishing",
        "Window cleaning exterior",
    ],
    "Security": [
        "Security guard services - {month}",
        "Access control system maintenance",
        "Security camera installation",
    ],
    "Medical Supplies": [
        "Disposable gloves {qty} boxes",
        "Surgical masks N95 {qty} cases",
        "IV administration sets {qty} ea",
    ],
    "Trucking": [
        "Freight delivery {origin} to {dest}",
        "LTL shipment {qty} pallets",
        "Flatbed transport {qty} loads",
    ],
    "Fuel": [
        "Diesel fuel {qty} gallons",
        "Unleaded gasoline {qty} gallons",
    ],
    "Insurance": [
        "General liability premium Q{q}",
        "Workers comp premium Q{q}",
    ],
}

_STREETS = [
    "123 Main St", "456 Industrial Blvd", "789 Commerce Dr",
    "1010 Oak Ave", "222 Business Park Way", "555 Warehouse Ln",
    "888 Factory Rd", "333 Enterprise Ct", "777 Market St",
    "444 Trade Center Dr", "999 Service Rd", "111 Professional Pl",
]
_CITIES = [
    "Springfield, IL 62701", "Portland, OR 97201", "Austin, TX 78701",
    "Denver, CO 80201", "Charlotte, NC 28201", "Phoenix, AZ 85001",
    "Columbus, OH 43201", "Indianapolis, IN 46201", "San Jose, CA 95101",
    "Jacksonville, FL 32201", "Nashville, TN 37201", "Milwaukee, WI 53201",
]
_CONTACTS = [
    "John Smith", "Maria Garcia", "James Wilson", "Sarah Johnson",
    "Robert Chen", "Lisa Patel", "Michael Brown", "Jennifer Lee",
    "David Kim", "Amanda Torres", "Chris Anderson", "Rachel Nguyen",
    "Kevin Martinez", "Emily Taylor", "Brian Jackson", "Stephanie White",
]
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Contracted vendors (by index into _ALL_VENDORS)
_CONTRACT_COUNT = 20


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class SampleDataGenerator:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.invoice_counter = 0
        self.line_item_counter = 0
        self.receipt_counter = 0
        self.vendors: list[dict] = []
        self.invoices: list[dict] = []
        self.receipts: list[dict] = []
        self.contracts: list[dict] = []
        self.fraud_items: list[dict] = []

        # Vendor base prices (set during generation)
        self._vendor_base_prices: dict[str, float] = {}
        # Track vendor invoicing history for behaviour anomaly injection
        self._vendor_history: dict[str, list[dict]] = {}

    # ---- IDs -----------------------------------------------------------

    def _next_inv_id(self) -> str:
        self.invoice_counter += 1
        return f"INV-{self.invoice_counter:05d}"

    def _next_li_id(self) -> str:
        self.line_item_counter += 1
        return f"LI-{self.line_item_counter:06d}"

    def _next_receipt_id(self) -> str:
        self.receipt_counter += 1
        return f"REC-{self.receipt_counter:05d}"

    def _rand_date(self) -> date:
        return DATE_START + timedelta(days=int(self.rng.integers(0, DATE_RANGE_DAYS)))

    def _rand_date_in_month(self, year: int, month: int) -> date:
        from calendar import monthrange
        _, last = monthrange(year, month)
        day = int(self.rng.integers(1, last + 1))
        return date(year, month, day)

    # ---- Descriptions ---------------------------------------------------

    def _desc(self, category: str, inv_date: date) -> str:
        templates = _DESCRIPTIONS.get(category, _DESCRIPTIONS.get("Consulting", ["Services"]))
        tmpl = str(self.rng.choice(templates))
        return tmpl.format(
            qty=int(self.rng.integers(5, 500)),
            psi=self.rng.choice([3000, 4000, 5000]),
            size=self.rng.choice([4, 6, 8, 10, 12, 14]),
            month=_MONTHS[inv_date.month - 1],
            origin="Warehouse",
            dest="Jobsite",
            q=(inv_date.month - 1) // 3 + 1,
        )

    # ====================================================================
    # 1. Vendor master
    # ====================================================================

    def _generate_vendors(self) -> None:
        collusion_a_addr = f"{self.rng.choice(_STREETS)}, {self.rng.choice(_CITIES)}"
        collusion_b_bank = f"{self.rng.integers(1000, 9999)}"

        for i, (name, subcat, sector) in enumerate(_ALL_VENDORS):
            vid = f"V{i + 1:04d}"
            is_col_a = name in {n for n, _ in _COLLUSION_A}
            is_col_b = name in {n for n, _ in _COLLUSION_B}

            addr = collusion_a_addr if is_col_a else f"{self.rng.choice(_STREETS)}, {self.rng.choice(_CITIES)}"
            bank = collusion_b_bank if is_col_b else f"{self.rng.integers(1000, 9999)}"

            added_offset = int(self.rng.integers(0, 365 * 3))
            self.vendors.append({
                "vendor_id": vid,
                "vendor_name": name,
                "address": addr,
                "phone": f"({self.rng.integers(200,999)}) {self.rng.integers(200,999)}-{self.rng.integers(1000,9999)}",
                "tax_id": f"{self.rng.integers(10,99)}-{self.rng.integers(1000000,9999999)}",
                "bank_account_last4": str(bank),
                "contact_person": str(self.rng.choice(_CONTACTS)),
                "category": subcat,
                "sector": sector,
                "date_added": (date(2021, 1, 1) + timedelta(days=added_offset)).isoformat(),
            })

            # Set a base unit price for each vendor
            if sector == "construction":
                self._vendor_base_prices[vid] = float(self.rng.uniform(50, 500))
            elif sector == "services":
                self._vendor_base_prices[vid] = float(self.rng.uniform(100, 300))
            elif sector == "healthcare":
                self._vendor_base_prices[vid] = float(self.rng.uniform(20, 200))
            else:
                self._vendor_base_prices[vid] = float(self.rng.uniform(30, 150))

    # ====================================================================
    # 2. Clean invoices
    # ====================================================================

    def _generate_clean_invoices(self, target: int = 5000) -> None:
        per_vendor = target // len(self.vendors) + 1
        generated = 0

        for v in self.vendors:
            vid = v["vendor_id"]
            vname = v["vendor_name"]
            cat = v["category"]
            sector = v["sector"]
            base_price = self._vendor_base_prices[vid]
            is_seasonal = vname in _SEASONAL_NAMES
            is_recurring = sector in ("office", "services") and self.rng.random() < 0.6

            # How many invoices for this vendor
            if is_recurring:
                n_inv = int(self.rng.integers(14, 20))  # roughly monthly
            else:
                n_inv = int(self.rng.integers(3, per_vendor + 1))

            for _ in range(n_inv):
                if generated >= target:
                    return
                inv_date = self._rand_date()

                # Seasonal modulation
                if is_seasonal:
                    if cat == "Landscaping" and inv_date.month in (11, 12, 1, 2):
                        continue
                    if cat == "HVAC Maintenance" and inv_date.month in (5, 6, 7, 8):
                        if self.rng.random() < 0.7:
                            continue

                # Normal price drift: 1-3% per year
                months_elapsed = (inv_date.year - 2024) * 12 + inv_date.month - 1
                drift = 1 + (self.rng.uniform(0.01, 0.03) * months_elapsed / 12)
                unit_price = round(base_price * drift * self.rng.uniform(0.95, 1.05), 2)

                n_items = int(self.rng.choice([1, 1, 1, 2, 2, 3, 3, 4, 5, 7]))
                has_po = self.rng.random() < 0.60
                po = f"PO-{self.rng.integers(10000, 99999)}" if has_po else ""

                inv_id = self._next_inv_id()
                items_rows = []
                total = 0.0
                for li_idx in range(n_items):
                    qty = round(float(self.rng.uniform(1, 50)), 1)
                    li_total = round(unit_price * qty, 2)
                    total += li_total
                    items_rows.append({
                        "line_item_id": self._next_li_id(),
                        "description": self._desc(cat, inv_date),
                        "quantity": qty,
                        "unit_price": unit_price,
                        "line_total": li_total,
                    })

                pay_date = inv_date + timedelta(days=int(self.rng.integers(20, 60)))
                if pay_date > DATE_END:
                    pay_date = DATE_END

                inv = {
                    "invoice_id": inv_id,
                    "invoice_number": inv_id,
                    "vendor_id": vid,
                    "vendor_name": vname,
                    "invoice_date": inv_date.isoformat(),
                    "total_amount": round(total, 2),
                    "po_number": po,
                    "payment_date": pay_date.isoformat(),
                    "payment_status": "Paid",
                    "approved_by": str(self.rng.choice(_CONTACTS[:4])),
                }
                self.invoices.append(inv)
                self._vendor_history.setdefault(vid, []).append(inv)

                # Flatten line items into invoice CSV rows
                for li in items_rows:
                    inv_copy = dict(inv)
                    inv_copy.update({
                        "line_item_id": li["line_item_id"],
                        "line_item_description": li["description"],
                        "quantity": li["quantity"],
                        "unit_price": li["unit_price"],
                        "line_total": li["line_total"],
                    })
                    # Replace the dict in self.invoices on first item,
                    # append extras.  Actually, for CSV we want one row
                    # per line item.
                if n_items > 1:
                    # Remove the summary row, replace with per-line rows
                    self.invoices.pop()
                    for li in items_rows:
                        row = dict(inv)
                        row["line_item_id"] = li["line_item_id"]
                        row["line_item_description"] = li["description"]
                        row["quantity"] = li["quantity"]
                        row["unit_price"] = li["unit_price"]
                        row["line_total"] = li["line_total"]
                        self.invoices.append(row)
                else:
                    # Single-line: add line-item fields to the existing row
                    li = items_rows[0]
                    self.invoices[-1]["line_item_id"] = li["line_item_id"]
                    self.invoices[-1]["line_item_description"] = li["description"]
                    self.invoices[-1]["quantity"] = li["quantity"]
                    self.invoices[-1]["unit_price"] = li["unit_price"]
                    self.invoices[-1]["line_total"] = li["line_total"]

                # Goods receipt for material vendors
                if sector in ("construction", "healthcare", "transport"):
                    rec_date = inv_date + timedelta(days=int(self.rng.integers(-5, 3)))
                    if rec_date < DATE_START:
                        rec_date = DATE_START
                    self.receipts.append({
                        "receipt_id": self._next_receipt_id(),
                        "invoice_id": inv_id,
                        "vendor_id": vid,
                        "receipt_date": rec_date.isoformat(),
                        "received_by": str(self.rng.choice(_CONTACTS)),
                        "status": "Received",
                    })

                generated += 1

    # ====================================================================
    # 3. Contracts
    # ====================================================================

    def _generate_contracts(self) -> None:
        contracted_vendors = self.vendors[:_CONTRACT_COUNT]
        for v in contracted_vendors:
            vid = v["vendor_id"]
            bp = self._vendor_base_prices[vid]
            start = date(2024, 1, 1)
            end = date(2025, 12, 31)
            payment_terms = str(self.rng.choice(["Net 30", "2/10 Net 30", "Net 60"]))
            escalation = round(float(self.rng.uniform(2, 4)), 1)

            scope_items = [self._desc(v["category"], start) for _ in range(3)]

            self.contracts.append({
                "vendor_id": vid,
                "vendor_name": v["vendor_name"],
                "contract_start_date": start.isoformat(),
                "contract_end_date": end.isoformat(),
                "auto_renewal": bool(self.rng.random() < 0.5),
                "auto_renewal_notice_days": int(self.rng.choice([30, 60, 90])),
                "payment_terms": payment_terms,
                "rates": [{
                    "item_description": scope_items[0],
                    "unit": str(self.rng.choice(["hour", "each", "LF", "CY", "ton"])),
                    "rate": str(round(bp, 2)),
                }],
                "volume_discounts": [{
                    "threshold_quantity": int(self.rng.choice([50, 100, 200])),
                    "discount_pct": str(round(float(self.rng.uniform(3, 8)), 1)),
                }],
                "scope_of_work": scope_items,
                "annual_escalation_pct": str(escalation),
            })

    # ====================================================================
    # FRAUD INJECTION
    # ====================================================================

    def _inject_all_fraud(self) -> None:
        self._inject_duplicate_invoices()
        self._inject_price_creep()
        self._inject_phantom_services()
        self._inject_vendor_collusion()
        self._inject_contract_violations()
        self._inject_vendor_behavior_anomalies()
        self._inject_split_invoicing()

    # ---- Fraud 1: Duplicate Invoices (15 pairs) -----------------------

    def _inject_duplicate_invoices(self) -> None:
        # Pick source invoices from the existing clean set
        unique_invs = {}
        for row in self.invoices:
            unique_invs.setdefault(row["invoice_id"], row)
        source_list = list(unique_invs.values())
        self.rng.shuffle(source_list)

        # 5 exact duplicates
        for src in source_list[:5]:
            dup = dict(src)
            # Same invoice_number — exact duplicate
            dup["invoice_id"] = src["invoice_id"]  # keep same ID
            dup["line_item_id"] = self._next_li_id()
            self.invoices.append(dup)
            self.fraud_items.append({
                "fraud_type": "duplicate_invoice",
                "subtype": "exact_duplicate",
                "description": f"Exact duplicate of invoice {src['invoice_number']}",
                "invoice_ids": [src["invoice_id"]],
                "vendor": src["vendor_name"],
                "amount_at_risk": src["total_amount"],
                "difficulty": "easy",
            })

        # 5 near-duplicates
        for src in source_list[5:10]:
            dup_id = self._next_inv_id()
            dup = dict(src)
            dup["invoice_id"] = dup_id
            dup["invoice_number"] = dup_id
            dup["line_item_id"] = self._next_li_id()
            # Amount within 2%
            factor = 1 + self.rng.uniform(-0.02, 0.02)
            dup["total_amount"] = round(src["total_amount"] * factor, 2)
            dup["line_total"] = dup["total_amount"]
            # Date within 7 days
            src_date = date.fromisoformat(src["invoice_date"])
            dup_date = src_date + timedelta(days=int(self.rng.integers(1, 7)))
            dup["invoice_date"] = dup_date.isoformat()
            self.invoices.append(dup)
            self.fraud_items.append({
                "fraud_type": "duplicate_invoice",
                "subtype": "near_duplicate",
                "description": f"Near-duplicate of {src['invoice_number']} — similar amount, "
                               f"{(dup_date - src_date).days} days apart",
                "invoice_ids": [src["invoice_id"], dup_id],
                "vendor": src["vendor_name"],
                "amount_at_risk": src["total_amount"],
                "difficulty": "medium",
            })

        # 3 reformatted duplicates
        for src in source_list[10:13]:
            dup_id = self._next_inv_id()
            dup = dict(src)
            dup["invoice_id"] = dup_id
            dup["invoice_number"] = f"RF-{self.rng.integers(10000, 99999)}"
            dup["line_item_id"] = self._next_li_id()
            # Same amount, different date within 30 days
            src_date = date.fromisoformat(src["invoice_date"])
            dup_date = src_date + timedelta(days=int(self.rng.integers(5, 30)))
            dup["invoice_date"] = dup_date.isoformat()
            self.invoices.append(dup)
            self.fraud_items.append({
                "fraud_type": "duplicate_invoice",
                "subtype": "reformatted_duplicate",
                "description": f"Reformatted duplicate of {src['invoice_number']} with different "
                               f"invoice number format",
                "invoice_ids": [src["invoice_id"], dup_id],
                "vendor": src["vendor_name"],
                "amount_at_risk": src["total_amount"],
                "difficulty": "hard",
            })

        # 2 cross-PO duplicates
        for src in source_list[13:15]:
            if not src.get("po_number"):
                src["po_number"] = f"PO-{self.rng.integers(10000, 99999)}"
            dup_id = self._next_inv_id()
            dup = dict(src)
            dup["invoice_id"] = dup_id
            dup["invoice_number"] = dup_id
            dup["line_item_id"] = self._next_li_id()
            dup["po_number"] = f"PO-{self.rng.integers(10000, 99999)}"
            self.invoices.append(dup)
            self.fraud_items.append({
                "fraud_type": "duplicate_invoice",
                "subtype": "cross_po_duplicate",
                "description": f"Same work billed under different POs: "
                               f"{src['po_number']} and {dup['po_number']}",
                "invoice_ids": [src["invoice_id"], dup_id],
                "vendor": src["vendor_name"],
                "amount_at_risk": src["total_amount"],
                "difficulty": "hard",
            })

    # ---- Fraud 2: Price Creep (8 vendors) -----------------------------

    def _inject_price_creep(self) -> None:
        # Pick 8 vendors with enough history
        vendor_ids_with_history = [
            vid for vid, hist in self._vendor_history.items()
            if len(hist) >= 6
        ]
        self.rng.shuffle(vendor_ids_with_history)
        creep_vendors = vendor_ids_with_history[:8]

        # 3 gradual (2-3% per quarter)
        for vid in creep_vendors[:3]:
            self._apply_price_creep(vid, "gradual", quarterly_pct=self.rng.uniform(2, 3))

        # 2 sudden jumps (8-12%)
        for vid in creep_vendors[3:5]:
            self._apply_price_creep(vid, "sudden_jump", jump_pct=self.rng.uniform(8, 12))

        # 2 accelerating
        for vid in creep_vendors[5:7]:
            self._apply_price_creep(vid, "accelerating")

        # 1 exceeding contracted rate by 15%
        if creep_vendors[7:]:
            self._apply_price_creep(creep_vendors[7], "contract_breach", breach_pct=15)

    def _apply_price_creep(self, vid: str, pattern: str, **kwargs) -> None:
        hist = self._vendor_history.get(vid, [])
        if not hist:
            return
        vname = hist[0]["vendor_name"]
        # Sort by date
        hist.sort(key=lambda x: x["invoice_date"])
        total_increase = 0.0

        for i, inv in enumerate(hist):
            # Find corresponding rows in self.invoices
            if pattern == "gradual":
                qtr = i // (len(hist) // 6 + 1)
                pct = kwargs.get("quarterly_pct", 2.5)
                increase = 1 + (pct / 100) * qtr
            elif pattern == "sudden_jump":
                midpoint = len(hist) // 2
                pct = kwargs.get("jump_pct", 10)
                increase = 1 + (pct / 100) if i >= midpoint else 1.0
            elif pattern == "accelerating":
                stage = i // max(len(hist) // 4, 1)
                rates = [0, 1, 2, 4, 6]
                increase = 1 + rates[min(stage, len(rates) - 1)] / 100
            elif pattern == "contract_breach":
                increase = 1 + kwargs.get("breach_pct", 15) / 100
            else:
                increase = 1.0

            if increase > 1.0:
                for row in self.invoices:
                    if row["invoice_id"] == inv["invoice_id"]:
                        old_price = row.get("unit_price", 0)
                        if old_price:
                            row["unit_price"] = round(float(old_price) * increase, 2)
                            row["line_total"] = round(row["unit_price"] * float(row.get("quantity", 1)), 2)
                            row["total_amount"] = row["line_total"]
                total_increase = max(total_increase, (increase - 1) * 100)

        if total_increase > 0:
            self.fraud_items.append({
                "fraud_type": "price_creep",
                "subtype": pattern,
                "description": f"Price creep on vendor {vname}: {pattern} pattern, "
                               f"up to {total_increase:.1f}% increase",
                "invoice_ids": [h["invoice_id"] for h in hist],
                "vendor": vname,
                "amount_at_risk": round(sum(
                    float(r["total_amount"]) * (total_increase / 100)
                    for r in hist if r.get("total_amount")
                ) / max(len(hist), 1), 2),
                "difficulty": "easy" if pattern == "sudden_jump" else "hard",
            })

    # ---- Fraud 3: Phantom Services (12 line items) --------------------

    def _inject_phantom_services(self) -> None:
        # 4 vague descriptions with no PO
        vague_descs = [
            "Consulting services", "Miscellaneous charges",
            "Professional fees", "Other charges",
        ]
        for desc in vague_descs:
            v = self.vendors[int(self.rng.integers(0, len(self.vendors)))]
            inv_id = self._next_inv_id()
            amt = round(float(self.rng.uniform(2000, 15000)), 2)
            inv_date = self._rand_date()
            row = {
                "invoice_id": inv_id, "invoice_number": inv_id,
                "vendor_id": v["vendor_id"], "vendor_name": v["vendor_name"],
                "invoice_date": inv_date.isoformat(),
                "total_amount": amt, "po_number": "",
                "payment_date": (inv_date + timedelta(days=30)).isoformat(),
                "payment_status": "Paid",
                "approved_by": str(self.rng.choice(_CONTACTS[:4])),
                "line_item_id": self._next_li_id(),
                "line_item_description": desc,
                "quantity": 1, "unit_price": amt, "line_total": amt,
            }
            self.invoices.append(row)
            self.fraud_items.append({
                "fraud_type": "phantom_services",
                "subtype": "vague_description",
                "description": f'Vague line item "{desc}" with no PO from {v["vendor_name"]}',
                "invoice_ids": [inv_id],
                "vendor": v["vendor_name"],
                "amount_at_risk": amt,
                "difficulty": "medium",
            })

        # 3 category mismatches
        mismatch_pairs = [
            ("Office Supplies", "Engineering consultation"),
            ("Cleaning", "Software development services"),
            ("Fuel", "Architecture design review"),
        ]
        for target_cat, fake_desc in mismatch_pairs:
            candidates = [v for v in self.vendors if v["category"] == target_cat]
            if not candidates:
                continue
            v = candidates[0]
            inv_id = self._next_inv_id()
            amt = round(float(self.rng.uniform(3000, 20000)), 2)
            inv_date = self._rand_date()
            row = {
                "invoice_id": inv_id, "invoice_number": inv_id,
                "vendor_id": v["vendor_id"], "vendor_name": v["vendor_name"],
                "invoice_date": inv_date.isoformat(),
                "total_amount": amt, "po_number": "",
                "payment_date": (inv_date + timedelta(days=30)).isoformat(),
                "payment_status": "Paid",
                "approved_by": str(self.rng.choice(_CONTACTS[:4])),
                "line_item_id": self._next_li_id(),
                "line_item_description": fake_desc,
                "quantity": 1, "unit_price": amt, "line_total": amt,
            }
            self.invoices.append(row)
            self.fraud_items.append({
                "fraud_type": "phantom_services",
                "subtype": "category_mismatch",
                "description": f'{v["vendor_name"]} ({target_cat} vendor) billing for '
                               f'"{fake_desc}"',
                "invoice_ids": [inv_id],
                "vendor": v["vendor_name"],
                "amount_at_risk": amt,
                "difficulty": "medium",
            })

        # 3 one-off charges
        for _ in range(3):
            v = self.vendors[int(self.rng.integers(0, 30))]
            inv_id = self._next_inv_id()
            amt = round(float(self.rng.uniform(5000, 25000)), 2)
            inv_date = self._rand_date()
            row = {
                "invoice_id": inv_id, "invoice_number": inv_id,
                "vendor_id": v["vendor_id"], "vendor_name": v["vendor_name"],
                "invoice_date": inv_date.isoformat(),
                "total_amount": amt,
                "po_number": f"PO-{self.rng.integers(10000, 99999)}",
                "payment_date": (inv_date + timedelta(days=30)).isoformat(),
                "payment_status": "Paid",
                "approved_by": str(self.rng.choice(_CONTACTS[:4])),
                "line_item_id": self._next_li_id(),
                "line_item_description": "Special project advisory services - one-time engagement",
                "quantity": 1, "unit_price": amt, "line_total": amt,
            }
            self.invoices.append(row)
            self.fraud_items.append({
                "fraud_type": "phantom_services",
                "subtype": "one_off_charge",
                "description": f'One-off charge from {v["vendor_name"]} outside normal category',
                "invoice_ids": [inv_id],
                "vendor": v["vendor_name"],
                "amount_at_risk": amt,
                "difficulty": "hard",
            })

        # 2 no matching goods receipt
        for _ in range(2):
            v = self.vendors[int(self.rng.integers(0, 25))]  # construction vendors
            inv_id = self._next_inv_id()
            amt = round(float(self.rng.uniform(8000, 30000)), 2)
            inv_date = self._rand_date()
            row = {
                "invoice_id": inv_id, "invoice_number": inv_id,
                "vendor_id": v["vendor_id"], "vendor_name": v["vendor_name"],
                "invoice_date": inv_date.isoformat(),
                "total_amount": amt,
                "po_number": f"PO-{self.rng.integers(10000, 99999)}",
                "payment_date": (inv_date + timedelta(days=30)).isoformat(),
                "payment_status": "Paid",
                "approved_by": str(self.rng.choice(_CONTACTS[:4])),
                "line_item_id": self._next_li_id(),
                "line_item_description": f"Material delivery - {self._desc(v['category'], inv_date)}",
                "quantity": int(self.rng.integers(10, 100)),
                "unit_price": round(amt / 50, 2),
                "line_total": amt,
            }
            self.invoices.append(row)
            # Deliberately do NOT create a goods receipt
            self.fraud_items.append({
                "fraud_type": "phantom_services",
                "subtype": "no_goods_receipt",
                "description": f"Material delivery invoice from {v['vendor_name']} with no "
                               f"matching goods receipt",
                "invoice_ids": [inv_id],
                "vendor": v["vendor_name"],
                "amount_at_risk": amt,
                "difficulty": "easy",
            })

    # ---- Fraud 4: Vendor Collusion (2 clusters) ----------------------

    def _inject_vendor_collusion(self) -> None:
        # Cluster A: 3 vendors same address, same-day invoicing, 30% above market
        col_a_vids = [v["vendor_id"] for v in self.vendors
                      if v["vendor_name"] in {n for n, _ in _COLLUSION_A}]
        cluster_a_ids = []
        for month_offset in range(0, 18, 3):
            inv_date = DATE_START + timedelta(days=month_offset * 30 + int(self.rng.integers(0, 5)))
            for vid in col_a_vids:
                v = next(vv for vv in self.vendors if vv["vendor_id"] == vid)
                inv_id = self._next_inv_id()
                base = float(self.rng.uniform(8000, 20000))
                amt = round(base * 1.30, 2)  # 30% above market
                row = {
                    "invoice_id": inv_id, "invoice_number": inv_id,
                    "vendor_id": vid, "vendor_name": v["vendor_name"],
                    "invoice_date": inv_date.isoformat(),
                    "total_amount": amt,
                    "po_number": f"PO-{self.rng.integers(10000, 99999)}",
                    "payment_date": (inv_date + timedelta(days=30)).isoformat(),
                    "payment_status": "Paid",
                    "approved_by": str(self.rng.choice(_CONTACTS[:2])),
                    "line_item_id": self._next_li_id(),
                    "line_item_description": self._desc(v["category"], inv_date),
                    "quantity": int(self.rng.integers(5, 30)),
                    "unit_price": round(amt / 10, 2),
                    "line_total": amt,
                }
                self.invoices.append(row)
                cluster_a_ids.append(inv_id)

        self.fraud_items.append({
            "fraud_type": "vendor_collusion",
            "subtype": "shared_address_cluster",
            "description": "Cluster A: 3 vendors share address, invoice on same days, "
                           "30% above market rate",
            "invoice_ids": cluster_a_ids,
            "vendor": ", ".join(n for n, _ in _COLLUSION_A),
            "amount_at_risk": round(sum(
                float(r["total_amount"]) for r in self.invoices
                if r["invoice_id"] in cluster_a_ids
            ) * 0.30, 2),
            "difficulty": "medium",
        })

        # Cluster B: 2 vendors same bank, rotating wins, 20% above market
        col_b_vids = [v["vendor_id"] for v in self.vendors
                      if v["vendor_name"] in {n for n, _ in _COLLUSION_B}]
        cluster_b_ids = []
        for month_offset in range(0, 18, 2):
            # Rotate which vendor "wins"
            winner_vid = col_b_vids[month_offset % 2]
            v = next(vv for vv in self.vendors if vv["vendor_id"] == winner_vid)
            inv_id = self._next_inv_id()
            inv_date = DATE_START + timedelta(days=month_offset * 30 + int(self.rng.integers(0, 10)))
            amt = round(float(self.rng.uniform(5000, 15000)) * 1.20, 2)
            row = {
                "invoice_id": inv_id, "invoice_number": inv_id,
                "vendor_id": winner_vid, "vendor_name": v["vendor_name"],
                "invoice_date": inv_date.isoformat(),
                "total_amount": amt,
                "po_number": f"PO-{self.rng.integers(10000, 99999)}",
                "payment_date": (inv_date + timedelta(days=30)).isoformat(),
                "payment_status": "Paid",
                "approved_by": str(self.rng.choice(_CONTACTS[:2])),
                "line_item_id": self._next_li_id(),
                "line_item_description": self._desc(v["category"], inv_date),
                "quantity": int(self.rng.integers(5, 20)),
                "unit_price": round(amt / 10, 2),
                "line_total": amt,
            }
            self.invoices.append(row)
            cluster_b_ids.append(inv_id)

        self.fraud_items.append({
            "fraud_type": "vendor_collusion",
            "subtype": "shared_bank_rotating",
            "description": "Cluster B: 2 vendors share bank account, rotate project wins, "
                           "20% above market",
            "invoice_ids": cluster_b_ids,
            "vendor": ", ".join(n for n, _ in _COLLUSION_B),
            "amount_at_risk": round(sum(
                float(r["total_amount"]) for r in self.invoices
                if r["invoice_id"] in cluster_b_ids
            ) * 0.20, 2),
            "difficulty": "hard",
        })

    # ---- Fraud 5: Contract Compliance Violations (10) -----------------

    def _inject_contract_violations(self) -> None:
        if len(self.contracts) < 10:
            return

        # 3 rate overcharges
        for c in self.contracts[:3]:
            vid = c["vendor_id"]
            contracted_rate = float(c["rates"][0]["rate"])
            inv_id = self._next_inv_id()
            overcharge_rate = round(contracted_rate * 1.15, 2)
            qty = int(self.rng.integers(10, 50))
            amt = round(overcharge_rate * qty, 2)
            inv_date = self._rand_date()
            v = next(vv for vv in self.vendors if vv["vendor_id"] == vid)
            row = self._make_fraud_invoice(inv_id, v, inv_date, amt,
                                           c["rates"][0]["item_description"],
                                           qty, overcharge_rate)
            self.invoices.append(row)
            self.fraud_items.append({
                "fraud_type": "contract_compliance",
                "subtype": "rate_overcharge",
                "description": f"Billing at ${overcharge_rate}/unit vs contracted "
                               f"${contracted_rate}/unit",
                "invoice_ids": [inv_id],
                "vendor": v["vendor_name"],
                "amount_at_risk": round((overcharge_rate - contracted_rate) * qty, 2),
                "difficulty": "easy",
            })

        # 2 missed early payment discounts
        for c in self.contracts[3:5]:
            if "2/10" not in c["payment_terms"]:
                c["payment_terms"] = "2/10 Net 30"
            vid = c["vendor_id"]
            v = next(vv for vv in self.vendors if vv["vendor_id"] == vid)
            inv_id = self._next_inv_id()
            amt = round(float(self.rng.uniform(5000, 20000)), 2)
            inv_date = self._rand_date()
            pay_date = inv_date + timedelta(days=8)  # within discount window
            row = self._make_fraud_invoice(inv_id, v, inv_date, amt,
                                           "Services per contract")
            row["payment_date"] = pay_date.isoformat()
            self.invoices.append(row)
            discount = round(amt * 0.02, 2)
            self.fraud_items.append({
                "fraud_type": "contract_compliance",
                "subtype": "missed_discount",
                "description": f"Paid within 10 days but 2% discount (${discount}) not applied",
                "invoice_ids": [inv_id],
                "vendor": v["vendor_name"],
                "amount_at_risk": discount,
                "difficulty": "easy",
            })

        # 2 out-of-scope billing
        for c in self.contracts[5:7]:
            vid = c["vendor_id"]
            v = next(vv for vv in self.vendors if vv["vendor_id"] == vid)
            inv_id = self._next_inv_id()
            amt = round(float(self.rng.uniform(3000, 12000)), 2)
            inv_date = self._rand_date()
            row = self._make_fraud_invoice(inv_id, v, inv_date, amt,
                                           "Strategic marketing campaign development")
            self.invoices.append(row)
            self.fraud_items.append({
                "fraud_type": "contract_compliance",
                "subtype": "out_of_scope",
                "description": f"Service not in contract scope for {v['vendor_name']}",
                "invoice_ids": [inv_id],
                "vendor": v["vendor_name"],
                "amount_at_risk": amt,
                "difficulty": "medium",
            })

        # 1 billing after contract expiration
        c = self.contracts[7]
        c["contract_end_date"] = "2024-09-30"
        vid = c["vendor_id"]
        v = next(vv for vv in self.vendors if vv["vendor_id"] == vid)
        inv_id = self._next_inv_id()
        amt = round(float(self.rng.uniform(5000, 15000)), 2)
        row = self._make_fraud_invoice(inv_id, v, date(2025, 1, 15), amt,
                                       c["rates"][0]["item_description"])
        self.invoices.append(row)
        self.fraud_items.append({
            "fraud_type": "contract_compliance",
            "subtype": "expired_contract",
            "description": f"Invoice dated 2025-01-15 but contract expired 2024-09-30",
            "invoice_ids": [inv_id],
            "vendor": v["vendor_name"],
            "amount_at_risk": amt,
            "difficulty": "easy",
        })

        # 1 excess escalation (7% vs allowed 3%)
        c = self.contracts[8]
        vid = c["vendor_id"]
        v = next(vv for vv in self.vendors if vv["vendor_id"] == vid)
        contracted_rate = float(c["rates"][0]["rate"])
        escalated_rate = round(contracted_rate * 1.07, 2)
        allowed_rate = round(contracted_rate * 1.03, 2)
        inv_id = self._next_inv_id()
        qty = int(self.rng.integers(20, 60))
        amt = round(escalated_rate * qty, 2)
        row = self._make_fraud_invoice(inv_id, v, date(2025, 3, 1), amt,
                                       c["rates"][0]["item_description"],
                                       qty, escalated_rate)
        self.invoices.append(row)
        self.fraud_items.append({
            "fraud_type": "contract_compliance",
            "subtype": "excess_escalation",
            "description": f"Applied 7% escalation vs contracted max "
                           f"{c['annual_escalation_pct']}%",
            "invoice_ids": [inv_id],
            "vendor": v["vendor_name"],
            "amount_at_risk": round((escalated_rate - allowed_rate) * qty, 2),
            "difficulty": "medium",
        })

        # 1 volume discount not applied
        c = self.contracts[9]
        vid = c["vendor_id"]
        v = next(vv for vv in self.vendors if vv["vendor_id"] == vid)
        contracted_rate = float(c["rates"][0]["rate"])
        disc_pct = float(c["volume_discounts"][0]["discount_pct"])
        threshold_qty = int(c["volume_discounts"][0]["threshold_quantity"])
        qty = threshold_qty + int(self.rng.integers(10, 50))
        amt = round(contracted_rate * qty, 2)  # no discount applied
        inv_id = self._next_inv_id()
        row = self._make_fraud_invoice(inv_id, v, self._rand_date(), amt,
                                       c["rates"][0]["item_description"],
                                       qty, contracted_rate)
        self.invoices.append(row)
        missed_disc = round(amt * disc_pct / 100, 2)
        self.fraud_items.append({
            "fraud_type": "contract_compliance",
            "subtype": "volume_discount_not_applied",
            "description": f"Ordered {qty} units (threshold {threshold_qty}), "
                           f"{disc_pct}% discount not applied",
            "invoice_ids": [inv_id],
            "vendor": v["vendor_name"],
            "amount_at_risk": missed_disc,
            "difficulty": "easy",
        })

    # ---- Fraud 6: Vendor Behavior Anomalies (5 vendors) ---------------

    def _inject_vendor_behavior_anomalies(self) -> None:
        vendor_ids_with_hist = [
            vid for vid, h in self._vendor_history.items() if len(h) >= 6
        ]
        self.rng.shuffle(vendor_ids_with_hist)
        targets = vendor_ids_with_hist[:5]

        # 1. Bank account change (BEC)
        if len(targets) >= 1:
            vid = targets[0]
            v = next(vv for vv in self.vendors if vv["vendor_id"] == vid)
            old_bank = v["bank_account_last4"]
            v["bank_account_last4"] = f"{self.rng.integers(1000, 9999)}"
            inv_id = self._next_inv_id()
            amt = round(float(self.rng.uniform(15000, 50000)), 2)
            row = self._make_fraud_invoice(inv_id, v, date(2025, 5, 1), amt,
                                           "Quarterly service fee")
            self.invoices.append(row)
            self.fraud_items.append({
                "fraud_type": "vendor_behavior",
                "subtype": "bank_account_change",
                "description": f"Bank account changed from *{old_bank} to "
                               f"*{v['bank_account_last4']} — potential BEC attack",
                "invoice_ids": [inv_id],
                "vendor": v["vendor_name"],
                "amount_at_risk": amt,
                "difficulty": "easy",
            })

        # 2. Sudden amount spike
        if len(targets) >= 2:
            vid = targets[1]
            v = next(vv for vv in self.vendors if vv["vendor_id"] == vid)
            inv_id = self._next_inv_id()
            amt = 45000.00
            row = self._make_fraud_invoice(inv_id, v, date(2025, 4, 15), amt,
                                           "Emergency equipment purchase")
            self.invoices.append(row)
            self.fraud_items.append({
                "fraud_type": "vendor_behavior",
                "subtype": "amount_spike",
                "description": f"Sudden spike to ${amt:,.2f} from vendor averaging ~$5,000",
                "invoice_ids": [inv_id],
                "vendor": v["vendor_name"],
                "amount_at_risk": amt,
                "difficulty": "easy",
            })

        # 3. Frequency change (monthly → weekly)
        if len(targets) >= 3:
            vid = targets[2]
            v = next(vv for vv in self.vendors if vv["vendor_id"] == vid)
            freq_ids = []
            for week in range(8):
                inv_id = self._next_inv_id()
                inv_date = date(2025, 4, 1) + timedelta(weeks=week)
                if inv_date > DATE_END:
                    break
                amt = round(float(self.rng.uniform(2000, 5000)), 2)
                row = self._make_fraud_invoice(inv_id, v, inv_date, amt,
                                               "Recurring services")
                self.invoices.append(row)
                freq_ids.append(inv_id)
            self.fraud_items.append({
                "fraud_type": "vendor_behavior",
                "subtype": "frequency_change",
                "description": f"Invoicing changed from monthly to weekly for {v['vendor_name']}",
                "invoice_ids": freq_ids,
                "vendor": v["vendor_name"],
                "amount_at_risk": round(sum(
                    float(r["total_amount"]) for r in self.invoices
                    if r["invoice_id"] in freq_ids
                ) * 0.5, 2),
                "difficulty": "medium",
            })

        # 4. Multiple anomalies (contact + email domain + format change)
        if len(targets) >= 4:
            vid = targets[3]
            v = next(vv for vv in self.vendors if vv["vendor_id"] == vid)
            v["contact_person"] = "Unknown Person"
            inv_id = self._next_inv_id()
            amt = round(float(self.rng.uniform(10000, 30000)), 2)
            row = self._make_fraud_invoice(inv_id, v, date(2025, 5, 10), amt,
                                           "Expedited services - new engagement")
            row["invoice_number"] = f"NEW-FMT-{self.rng.integers(1000,9999)}"
            self.invoices.append(row)
            self.fraud_items.append({
                "fraud_type": "vendor_behavior",
                "subtype": "multiple_anomalies",
                "description": f"Contact person changed, invoice format changed, "
                               f"unusual amount for {v['vendor_name']}",
                "invoice_ids": [inv_id],
                "vendor": v["vendor_name"],
                "amount_at_risk": amt,
                "difficulty": "medium",
            })

        # 5. Gradual behavioural drift over 6 months
        if len(targets) >= 5:
            vid = targets[4]
            v = next(vv for vv in self.vendors if vv["vendor_id"] == vid)
            drift_ids = []
            base_amt = 3000.0
            for m in range(6):
                inv_id = self._next_inv_id()
                inv_date = date(2025, 1 + m, 15)
                drift_factor = 1 + (m * 0.08)  # 8% per month drift
                amt = round(base_amt * drift_factor, 2)
                row = self._make_fraud_invoice(inv_id, v, inv_date, amt,
                                               "Regular maintenance services")
                self.invoices.append(row)
                drift_ids.append(inv_id)
            self.fraud_items.append({
                "fraud_type": "vendor_behavior",
                "subtype": "gradual_drift",
                "description": f"Amount drifted from ${base_amt:,.2f} to "
                               f"${amt:,.2f} over 6 months for {v['vendor_name']}",
                "invoice_ids": drift_ids,
                "vendor": v["vendor_name"],
                "amount_at_risk": round(amt - base_amt, 2),
                "difficulty": "hard",
            })

    # ---- Fraud 7: Split Invoicing (4 patterns) -----------------------

    def _inject_split_invoicing(self) -> None:
        # 2 threshold clustering patterns ($4,800-$4,999 below $5K threshold)
        for _ in range(2):
            v = self.vendors[int(self.rng.integers(0, len(self.vendors)))]
            cluster_ids = []
            for _ in range(6):
                inv_id = self._next_inv_id()
                amt = round(float(self.rng.uniform(4800, 4999)), 2)
                inv_date = self._rand_date()
                row = self._make_fraud_invoice(inv_id, v, inv_date, amt,
                                               self._desc(v["category"], inv_date))
                self.invoices.append(row)
                cluster_ids.append(inv_id)
            self.fraud_items.append({
                "fraud_type": "split_invoicing",
                "subtype": "threshold_clustering",
                "description": f"6 invoices from {v['vendor_name']} clustered at "
                               f"$4,800-$4,999 (below $5K approval threshold)",
                "invoice_ids": cluster_ids,
                "vendor": v["vendor_name"],
                "amount_at_risk": round(sum(
                    float(r["total_amount"]) for r in self.invoices
                    if r["invoice_id"] in cluster_ids
                ), 2),
                "difficulty": "medium",
            })

        # 1 temporal splitting ($30K split into 4 × ~$7,500 within 2 weeks)
        v = self.vendors[int(self.rng.integers(0, len(self.vendors)))]
        split_ids = []
        base_date = date(2025, 3, 1)
        for i in range(4):
            inv_id = self._next_inv_id()
            amt = round(float(self.rng.uniform(7200, 7800)), 2)
            inv_date = base_date + timedelta(days=int(self.rng.integers(0, 14)))
            po = f"PO-{self.rng.integers(10000, 99999)}"
            row = self._make_fraud_invoice(inv_id, v, inv_date, amt,
                                           "Project phase work", po_num=po)
            self.invoices.append(row)
            split_ids.append(inv_id)
        self.fraud_items.append({
            "fraud_type": "split_invoicing",
            "subtype": "temporal_split",
            "description": f"$30K job split into 4 invoices of ~$7,500 within 2 weeks "
                           f"from {v['vendor_name']}",
            "invoice_ids": split_ids,
            "vendor": v["vendor_name"],
            "amount_at_risk": round(sum(
                float(r["total_amount"]) for r in self.invoices
                if r["invoice_id"] in split_ids
            ), 2),
            "difficulty": "medium",
        })

        # 1 PO-linked splitting (3 invoices same PO, each below threshold)
        v = self.vendors[int(self.rng.integers(0, len(self.vendors)))]
        shared_po = f"PO-{self.rng.integers(10000, 99999)}"
        po_split_ids = []
        for _ in range(3):
            inv_id = self._next_inv_id()
            amt = round(float(self.rng.uniform(4500, 4950)), 2)
            inv_date = self._rand_date()
            row = self._make_fraud_invoice(inv_id, v, inv_date, amt,
                                           "Equipment and supplies",
                                           po_num=shared_po)
            self.invoices.append(row)
            po_split_ids.append(inv_id)
        self.fraud_items.append({
            "fraud_type": "split_invoicing",
            "subtype": "po_linked_split",
            "description": f"3 invoices referencing same PO {shared_po}, each below "
                           f"$5K threshold, from {v['vendor_name']}",
            "invoice_ids": po_split_ids,
            "vendor": v["vendor_name"],
            "amount_at_risk": round(sum(
                float(r["total_amount"]) for r in self.invoices
                if r["invoice_id"] in po_split_ids
            ), 2),
            "difficulty": "medium",
        })

    # ---- Helper -------------------------------------------------------

    def _make_fraud_invoice(
        self, inv_id: str, v: dict, inv_date: date, amt: float,
        description: str, qty: int = 1, unit_price: float = None,
        po_num: str = "",
    ) -> dict:
        if unit_price is None:
            unit_price = amt
        pay_date = inv_date + timedelta(days=int(self.rng.integers(20, 45)))
        if pay_date > DATE_END:
            pay_date = DATE_END
        return {
            "invoice_id": inv_id,
            "invoice_number": inv_id,
            "vendor_id": v["vendor_id"],
            "vendor_name": v["vendor_name"],
            "invoice_date": inv_date.isoformat(),
            "total_amount": round(amt, 2),
            "po_number": po_num,
            "payment_date": pay_date.isoformat(),
            "payment_status": "Paid",
            "approved_by": str(self.rng.choice(_CONTACTS[:4])),
            "line_item_id": self._next_li_id(),
            "line_item_description": description,
            "quantity": qty,
            "unit_price": round(unit_price, 2),
            "line_total": round(amt, 2),
        }

    # ====================================================================
    # Orchestrate & write
    # ====================================================================

    def generate(self, output_dir: str) -> None:
        out = Path(output_dir)

        logger.info("Generating vendors...")
        self._generate_vendors()

        logger.info("Generating clean invoices...")
        self._generate_clean_invoices(target=5000)

        logger.info("Generating contracts...")
        self._generate_contracts()

        logger.info("Injecting fraud patterns...")
        self._inject_all_fraud()

        logger.info("Writing output files...")
        self._write_invoices(out)
        self._write_vendors(out)
        self._write_contracts(out)
        self._write_receipts(out)
        self._write_approval_thresholds(out)
        self._write_fraud_key(out)

        total_risk = sum(
            float(f["amount_at_risk"]) for f in self.fraud_items
        )
        logger.info("=" * 60)
        logger.info("Generation complete:")
        logger.info("  Invoices (rows):  %d", len(self.invoices))
        logger.info("  Vendors:          %d", len(self.vendors))
        logger.info("  Contracts:        %d", len(self.contracts))
        logger.info("  Goods Receipts:   %d", len(self.receipts))
        logger.info("  Fraud items:      %d", len(self.fraud_items))
        logger.info("  Total at risk:    $%s", f"{total_risk:,.2f}")
        logger.info("=" * 60)

    def _write_invoices(self, out: Path) -> None:
        d = out / "sample_invoices"
        d.mkdir(parents=True, exist_ok=True)
        fields = [
            "invoice_id", "invoice_number", "vendor_id", "vendor_name",
            "invoice_date", "total_amount", "po_number", "payment_date",
            "payment_status", "approved_by", "line_item_id",
            "line_item_description", "quantity", "unit_price", "line_total",
        ]
        with open(d / "invoices.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.invoices)

    def _write_vendors(self, out: Path) -> None:
        d = out / "sample_vendor_master"
        d.mkdir(parents=True, exist_ok=True)
        fields = [
            "vendor_id", "vendor_name", "address", "phone", "tax_id",
            "bank_account_last4", "contact_person", "category", "date_added",
        ]
        with open(d / "vendors.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.vendors)

    def _write_contracts(self, out: Path) -> None:
        d = out / "sample_contracts"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "contracts.json", "w", encoding="utf-8") as f:
            json.dump(self.contracts, f, indent=2)

    def _write_receipts(self, out: Path) -> None:
        d = out / "sample_invoices"
        d.mkdir(parents=True, exist_ok=True)
        fields = ["receipt_id", "invoice_id", "vendor_id",
                   "receipt_date", "received_by", "status"]
        with open(d / "goods_receipts.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.receipts)

    def _write_approval_thresholds(self, out: Path) -> None:
        d = out / "sample_invoices"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "approval_thresholds.json", "w", encoding="utf-8") as f:
            json.dump(APPROVAL_THRESHOLDS, f, indent=2)

    def _write_fraud_key(self, out: Path) -> None:
        d = out / "sample_invoices"
        d.mkdir(parents=True, exist_ok=True)
        total_risk = round(sum(float(f["amount_at_risk"]) for f in self.fraud_items), 2)
        key = {
            "fraud_items": self.fraud_items,
            "total_fraud_count": len(self.fraud_items),
            "total_amount_at_risk": total_risk,
        }
        with open(d / "fraud_key.json", "w", encoding="utf-8") as f:
            json.dump(key, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sample invoice data with injected fraud patterns."
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Root output directory (default: data/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    gen = SampleDataGenerator(seed=args.seed)
    gen.generate(args.output_dir)


if __name__ == "__main__":
    main()
