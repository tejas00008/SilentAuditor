"""Date parsing and manipulation utilities for SilentAuditor."""

import logging
import re
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Ordered from most specific / unambiguous to most ambiguous.
# ISO 8601 first, then US, then European, then abbreviated month formats.
_DATE_FORMATS: list[str] = [
    # ISO / unambiguous
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%SZ",
    # US-style
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m/%d/%y",
    "%m-%d-%y",
    # European-style
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    # Abbreviated month
    "%d-%b-%Y",
    "%d-%b-%y",
    "%d %b %Y",
    "%d %b %y",
    "%b %d, %Y",
    "%b %d %Y",
    # Full month name
    "%B %d, %Y",
    "%B %d %Y",
    "%d %B %Y",
    "%d %B, %Y",
]


def parse_date(value: str) -> Optional[date]:
    """Parse a date string using multiple common formats.

    Tries ISO-8601, US (MM/DD/YYYY), European (DD/MM/YYYY), abbreviated
    month (DD-Mon-YY), and long-form (``January 15, 2025``) formats.

    Returns ``None`` when the value cannot be parsed — never raises.
    """
    if not value or not isinstance(value, str):
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    # Handle ordinal suffixes: "15th", "1st", "2nd", "3rd"
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", cleaned)

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue

    logger.debug("Unable to parse date string: %r", value)
    return None


def days_between(date1: date, date2: date) -> int:
    """Return the absolute number of days between two dates."""
    return abs((date2 - date1).days)


def months_between(date1: date, date2: date) -> int:
    """Return the absolute number of whole calendar months between two dates."""
    earlier, later = sorted((date1, date2))
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def is_weekend(d: date) -> bool:
    """Return ``True`` if *d* falls on Saturday or Sunday."""
    return d.weekday() >= 5


def is_business_day(d: date) -> bool:
    """Return ``True`` if *d* is a weekday (Mon–Fri).

    Does **not** account for public holidays.
    """
    return d.weekday() < 5


def get_quarter(d: date) -> str:
    """Return the quarter label, e.g. ``'2025-Q1'``."""
    quarter = (d.month - 1) // 3 + 1
    return f"{d.year}-Q{quarter}"


def date_to_period(d: date, period: str) -> str:
    """Convert a date to a period label.

    Args:
        d: The date to convert.
        period: One of ``'daily'``, ``'weekly'``, ``'monthly'``,
                ``'quarterly'``.

    Returns:
        A string label for the period (e.g. ``'2025-01-15'``,
        ``'2025-W03'``, ``'2025-01'``, ``'2025-Q1'``).

    Raises:
        ValueError: If *period* is not recognised.
    """
    period_lower = period.lower()
    if period_lower == "daily":
        return d.isoformat()
    if period_lower == "weekly":
        iso_year, iso_week, _ = d.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    if period_lower == "monthly":
        return f"{d.year}-{d.month:02d}"
    if period_lower == "quarterly":
        return get_quarter(d)
    raise ValueError(f"Unknown period type: {period!r}")
