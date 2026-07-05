"""Statistical utility functions for SilentAuditor detection modules."""

import logging
import math
from collections import Counter
from datetime import date
from decimal import Decimal
from typing import Optional

import numpy as np
from scipy import stats as sp_stats
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Benford's Law
# ------------------------------------------------------------------

def benfords_first_digit_expected() -> dict[int, float]:
    """Return the expected first-digit frequency distribution per Benford's Law.

    Keys are digits 1–9, values are probabilities that sum to 1.0.
    """
    return {d: math.log10(1 + 1 / d) for d in range(1, 10)}


def benfords_first_two_digits_expected() -> dict[int, float]:
    """Return the expected first-two-digit frequency distribution (10–99)."""
    return {d: math.log10(1 + 1 / d) for d in range(10, 100)}


def _extract_leading_digits(amounts: list[Decimal], n_digits: int) -> list[int]:
    """Extract the first *n_digits* digits from each positive amount."""
    result: list[int] = []
    for amt in amounts:
        abs_val = abs(amt)
        if abs_val == 0:
            continue
        s = str(abs_val).lstrip("0").lstrip(".").lstrip("0")
        digits_only = "".join(ch for ch in s if ch.isdigit())
        if len(digits_only) >= n_digits:
            result.append(int(digits_only[:n_digits]))
    return result


def benfords_test(
    amounts: list[Decimal], digits: int = 1
) -> dict:
    """Test a set of amounts against Benford's Law.

    Args:
        amounts: List of monetary values.
        digits: ``1`` for first-digit test, ``2`` for first-two-digits test.

    Returns:
        Dictionary with ``chi_squared``, ``p_value``, ``mad``,
        ``observed_distribution``, ``expected_distribution``, and
        ``significant`` (True when *p* < 0.05).
    """
    if digits not in (1, 2):
        raise ValueError("digits must be 1 or 2")

    leading = _extract_leading_digits(amounts, digits)

    if not leading:
        return {
            "chi_squared": 0.0,
            "p_value": 1.0,
            "mad": 0.0,
            "observed_distribution": {},
            "expected_distribution": {},
            "significant": False,
        }

    expected = (
        benfords_first_digit_expected()
        if digits == 1
        else benfords_first_two_digits_expected()
    )

    n = len(leading)
    counts = Counter(leading)

    observed_dist: dict[int, float] = {}
    for key in expected:
        observed_dist[key] = counts.get(key, 0) / n

    # Chi-squared statistic
    chi_sq = 0.0
    for key in expected:
        exp_count = expected[key] * n
        obs_count = counts.get(key, 0)
        if exp_count > 0:
            chi_sq += (obs_count - exp_count) ** 2 / exp_count

    dof = len(expected) - 1
    p_value = float(1.0 - sp_stats.chi2.cdf(chi_sq, dof))

    # Mean Absolute Deviation
    mad = sum(abs(observed_dist[k] - expected[k]) for k in expected) / len(expected)

    return {
        "chi_squared": chi_sq,
        "p_value": p_value,
        "mad": mad,
        "observed_distribution": observed_dist,
        "expected_distribution": expected,
        "significant": p_value < 0.05,
    }


# ------------------------------------------------------------------
# Basic statistics
# ------------------------------------------------------------------

def compute_zscore(value: float, mean: float, std: float) -> float:
    """Compute the z-score of *value* given a *mean* and *std*.

    Returns 0.0 when *std* is zero or negative.
    """
    if std <= 0:
        return 0.0
    return (value - mean) / std


def pearson_correlation(series_a: list[float], series_b: list[float]) -> float:
    """Compute the Pearson correlation coefficient between two series.

    Returns 0.0 when either series has fewer than 2 elements, or when
    either series has zero variance.
    """
    if len(series_a) < 2 or len(series_b) < 2:
        return 0.0
    if len(series_a) != len(series_b):
        raise ValueError("series_a and series_b must have the same length")

    a = np.array(series_a, dtype=np.float64)
    b = np.array(series_b, dtype=np.float64)

    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0

    r, _ = sp_stats.pearsonr(a, b)
    return float(r)


# ------------------------------------------------------------------
# Anomaly detection
# ------------------------------------------------------------------

def compute_isolation_forest_scores(data: np.ndarray) -> np.ndarray:
    """Fit an IsolationForest and return anomaly scores.

    Scores are in the range roughly [-1, 1]. Lower (more negative) values
    indicate stronger anomalies.  Returns an empty array when the input
    is empty or has fewer than 2 samples.
    """
    if data.size == 0 or (data.ndim >= 1 and data.shape[0] < 2):
        return np.array([], dtype=np.float64)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    model = IsolationForest(random_state=42, contamination="auto")
    model.fit(data)
    return model.decision_function(data)


# ------------------------------------------------------------------
# Regression / trend
# ------------------------------------------------------------------

def linear_regression_trend(
    dates: list[date], values: list[float]
) -> dict:
    """Fit a linear regression on (date-ordinal, value) pairs.

    Returns:
        Dictionary with ``slope``, ``intercept``, ``r_squared``,
        ``p_value``, and ``cumulative_change_pct`` (percentage change
        from fitted start to fitted end).
    """
    if len(dates) < 2 or len(values) < 2:
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "p_value": 1.0,
            "cumulative_change_pct": 0.0,
        }
    if len(dates) != len(values):
        raise ValueError("dates and values must have the same length")

    x = np.array([d.toordinal() for d in dates], dtype=np.float64)
    y = np.array(values, dtype=np.float64)

    slope, intercept, r_value, p_value, _ = sp_stats.linregress(x, y)

    fitted_start = slope * x[0] + intercept
    fitted_end = slope * x[-1] + intercept

    if fitted_start != 0:
        cumulative_change_pct = ((fitted_end - fitted_start) / abs(fitted_start)) * 100
    else:
        cumulative_change_pct = 0.0

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "cumulative_change_pct": float(cumulative_change_pct),
    }


# ------------------------------------------------------------------
# Chi-squared test (general purpose)
# ------------------------------------------------------------------

def chi_squared_test(observed: dict, expected: dict) -> dict:
    """Run a chi-squared goodness-of-fit test.

    *observed* and *expected* are dictionaries with the same keys.
    Values in *expected* are expected **counts** (not proportions).

    Returns ``chi_squared``, ``p_value``, and ``significant`` (p < 0.05).
    """
    if not observed or not expected:
        return {"chi_squared": 0.0, "p_value": 1.0, "significant": False}

    keys = sorted(set(observed) | set(expected))
    chi_sq = 0.0
    for k in keys:
        obs = observed.get(k, 0)
        exp = expected.get(k, 0)
        if exp > 0:
            chi_sq += (obs - exp) ** 2 / exp

    dof = max(len(keys) - 1, 1)
    p_value = float(1.0 - sp_stats.chi2.cdf(chi_sq, dof))

    return {
        "chi_squared": chi_sq,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


# ------------------------------------------------------------------
# Split-invoicing detection helper
# ------------------------------------------------------------------

def detect_clustering_below_threshold(
    amounts: list[Decimal],
    threshold: Decimal,
    band_pct: float = 0.10,
) -> dict:
    """Check whether amounts cluster just below a threshold.

    Looks at the band ``[threshold * (1 - band_pct), threshold)`` and
    compares the observed count to the expected count under a uniform
    distribution spanning ``[0, threshold)``.

    Returns:
        Dictionary with ``count_in_band``, ``expected_count``,
        ``chi_squared``, ``p_value``, and ``significant``.
    """
    if not amounts or threshold <= 0:
        return {
            "count_in_band": 0,
            "expected_count": 0.0,
            "chi_squared": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    lower_bound = threshold * Decimal(str(1 - band_pct))

    below_threshold = [a for a in amounts if Decimal("0") < a < threshold]
    if not below_threshold:
        return {
            "count_in_band": 0,
            "expected_count": 0.0,
            "chi_squared": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    count_in_band = sum(1 for a in below_threshold if lower_bound <= a < threshold)
    n = len(below_threshold)

    # Under uniform distribution, the expected proportion in the band
    expected_proportion = band_pct  # band width / total range
    expected_count = n * expected_proportion

    # 2-bin chi-squared: in-band vs out-of-band
    obs_in = count_in_band
    obs_out = n - count_in_band
    exp_in = expected_count
    exp_out = n - expected_count

    chi_sq = 0.0
    if exp_in > 0:
        chi_sq += (obs_in - exp_in) ** 2 / exp_in
    if exp_out > 0:
        chi_sq += (obs_out - exp_out) ** 2 / exp_out

    p_value = float(1.0 - sp_stats.chi2.cdf(chi_sq, 1))

    return {
        "count_in_band": count_in_band,
        "expected_count": float(expected_count),
        "chi_squared": float(chi_sq),
        "p_value": p_value,
        "significant": p_value < 0.05,
    }
