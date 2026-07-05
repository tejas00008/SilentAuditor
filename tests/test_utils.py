"""Tests for src/utils/ — date_utils, similarity, statistics."""

import math
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.utils.date_utils import (
    date_to_period,
    days_between,
    get_quarter,
    is_business_day,
    is_weekend,
    months_between,
    parse_date,
)
from src.utils.similarity import (
    batch_cosine_similarity,
    compute_embedding,
    cosine_similarity,
    exact_match,
    fuzzy_match_score,
    normalize_address,
    normalize_phone,
    normalize_text,
    normalize_vendor_name,
)
from src.utils.statistics import (
    benfords_first_digit_expected,
    benfords_first_two_digits_expected,
    benfords_test,
    chi_squared_test,
    compute_isolation_forest_scores,
    compute_zscore,
    detect_clustering_below_threshold,
    linear_regression_trend,
    pearson_correlation,
)


# ======================================================================
# date_utils
# ======================================================================

class TestParseDate:
    def test_iso_format(self):
        assert parse_date("2025-01-15") == date(2025, 1, 15)

    def test_us_format(self):
        assert parse_date("01/15/2025") == date(2025, 1, 15)

    def test_us_format_dash(self):
        assert parse_date("01-15-2025") == date(2025, 1, 15)

    def test_iso_with_slashes(self):
        assert parse_date("2025/01/15") == date(2025, 1, 15)

    def test_abbreviated_month(self):
        assert parse_date("15-Jan-2025") == date(2025, 1, 15)

    def test_abbreviated_month_short_year(self):
        assert parse_date("15-Jan-25") == date(2025, 1, 15)

    def test_long_month_name(self):
        assert parse_date("January 15, 2025") == date(2025, 1, 15)

    def test_long_month_no_comma(self):
        assert parse_date("January 15 2025") == date(2025, 1, 15)

    def test_ordinal_suffix(self):
        assert parse_date("January 15th, 2025") == date(2025, 1, 15)
        assert parse_date("January 1st, 2025") == date(2025, 1, 1)
        assert parse_date("January 2nd, 2025") == date(2025, 1, 2)
        assert parse_date("January 3rd, 2025") == date(2025, 1, 3)

    def test_iso_with_time(self):
        assert parse_date("2025-01-15T10:30:00") == date(2025, 1, 15)

    def test_iso_with_z(self):
        assert parse_date("2025-01-15T10:30:00Z") == date(2025, 1, 15)

    def test_unparseable_returns_none(self):
        assert parse_date("not-a-date") is None

    def test_empty_returns_none(self):
        assert parse_date("") is None

    def test_none_returns_none(self):
        assert parse_date(None) is None

    def test_whitespace_trimmed(self):
        assert parse_date("  2025-01-15  ") == date(2025, 1, 15)

    def test_european_format(self):
        # 31/12/2025 cannot be MM/DD since there is no month 31
        assert parse_date("31/12/2025") == date(2025, 12, 31)


class TestDaysBetween:
    def test_same_date(self):
        d = date(2025, 1, 1)
        assert days_between(d, d) == 0

    def test_one_day(self):
        assert days_between(date(2025, 1, 1), date(2025, 1, 2)) == 1

    def test_order_independent(self):
        d1, d2 = date(2025, 1, 1), date(2025, 6, 15)
        assert days_between(d1, d2) == days_between(d2, d1)


class TestMonthsBetween:
    def test_same_month(self):
        assert months_between(date(2025, 1, 1), date(2025, 1, 31)) == 0

    def test_one_month(self):
        assert months_between(date(2025, 1, 15), date(2025, 2, 15)) == 1

    def test_across_years(self):
        assert months_between(date(2024, 11, 1), date(2025, 2, 1)) == 3

    def test_order_independent(self):
        d1, d2 = date(2024, 1, 1), date(2025, 6, 1)
        assert months_between(d1, d2) == months_between(d2, d1)


class TestWeekendBusinessDay:
    def test_saturday(self):
        assert is_weekend(date(2025, 3, 29)) is True
        assert is_business_day(date(2025, 3, 29)) is False

    def test_sunday(self):
        assert is_weekend(date(2025, 3, 30)) is True

    def test_monday(self):
        assert is_weekend(date(2025, 3, 31)) is False
        assert is_business_day(date(2025, 3, 31)) is True

    def test_friday(self):
        assert is_weekend(date(2025, 3, 28)) is False
        assert is_business_day(date(2025, 3, 28)) is True


class TestGetQuarter:
    def test_q1(self):
        assert get_quarter(date(2025, 1, 15)) == "2025-Q1"
        assert get_quarter(date(2025, 3, 31)) == "2025-Q1"

    def test_q2(self):
        assert get_quarter(date(2025, 4, 1)) == "2025-Q2"

    def test_q3(self):
        assert get_quarter(date(2025, 7, 1)) == "2025-Q3"

    def test_q4(self):
        assert get_quarter(date(2025, 12, 31)) == "2025-Q4"


class TestDateToPeriod:
    def test_daily(self):
        assert date_to_period(date(2025, 1, 15), "daily") == "2025-01-15"

    def test_weekly(self):
        result = date_to_period(date(2025, 1, 15), "weekly")
        assert result.startswith("2025-W")

    def test_monthly(self):
        assert date_to_period(date(2025, 1, 15), "monthly") == "2025-01"

    def test_quarterly(self):
        assert date_to_period(date(2025, 1, 15), "quarterly") == "2025-Q1"

    def test_case_insensitive(self):
        assert date_to_period(date(2025, 1, 15), "Monthly") == "2025-01"

    def test_unknown_period_raises(self):
        with pytest.raises(ValueError, match="Unknown period"):
            date_to_period(date(2025, 1, 1), "yearly")


# ======================================================================
# similarity
# ======================================================================

class TestNormalizeText:
    def test_basic(self):
        assert normalize_text("  Hello, World!  ") == "hello world"

    def test_unicode(self):
        assert normalize_text("Café résumé") == "cafe resume"

    def test_empty(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""

    def test_collapses_whitespace(self):
        assert normalize_text("a   b\tc\nd") == "a b c d"

    def test_strips_punctuation(self):
        assert normalize_text("O'Brien & Co.") == "o brien co"


class TestNormalizeVendorName:
    def test_strips_inc(self):
        assert normalize_vendor_name("Acme Inc.") == "acme"

    def test_strips_llc(self):
        assert normalize_vendor_name("Widget LLC") == "widget"

    def test_strips_corporation(self):
        assert normalize_vendor_name("Global Corporation") == "global"

    def test_strips_ltd(self):
        assert normalize_vendor_name("Smith Ltd.") == "smith"

    def test_strips_co(self):
        assert normalize_vendor_name("Jones Co.") == "jones"

    def test_strips_gmbh(self):
        assert normalize_vendor_name("Bayern GmbH") == "bayern"

    def test_preserves_core_name(self):
        assert normalize_vendor_name("Advanced Micro Devices") == "advanced micro devices"

    def test_empty(self):
        assert normalize_vendor_name("") == ""

    def test_multiple_suffixes(self):
        # After stripping Co then Inc should also be handled
        result = normalize_vendor_name("Acme, Inc.")
        assert result == "acme"


class TestNormalizeAddress:
    def test_expands_abbreviations(self):
        assert "street" in normalize_address("123 Main St")

    def test_removes_suite(self):
        result = normalize_address("123 Main St Ste 400")
        assert "400" not in result
        assert "street" in result

    def test_removes_apt(self):
        result = normalize_address("456 Oak Ave Apt 2B")
        assert "2b" not in result.lower()

    def test_empty(self):
        assert normalize_address("") == ""

    def test_boulevard(self):
        assert "boulevard" in normalize_address("789 Sunset Blvd")


class TestNormalizePhone:
    def test_strips_formatting(self):
        assert normalize_phone("(555) 123-4567") == "5551234567"

    def test_strips_country_code(self):
        assert normalize_phone("+1-555-123-4567") == "5551234567"

    def test_already_clean(self):
        assert normalize_phone("5551234567") == "5551234567"

    def test_empty(self):
        assert normalize_phone("") == ""

    def test_international_kept(self):
        # 12-digit non-US number should be kept as-is
        assert normalize_phone("+44 20 7946 0958") == "442079460958"


class TestFuzzyMatchScore:
    def test_identical(self):
        assert fuzzy_match_score("hello", "hello") == 1.0

    def test_similar(self):
        score = fuzzy_match_score("Acme Corporation", "Acme Corp")
        assert score > 0.7

    def test_different(self):
        score = fuzzy_match_score("alpha", "zzzzz")
        assert score < 0.3

    def test_empty_returns_zero(self):
        assert fuzzy_match_score("", "hello") == 0.0
        assert fuzzy_match_score("hello", "") == 0.0

    def test_range(self):
        score = fuzzy_match_score("abc", "abd")
        assert 0.0 <= score <= 1.0


class TestExactMatch:
    def test_identical(self):
        assert exact_match("hello", "hello") is True

    def test_case_insensitive(self):
        assert exact_match("Hello", "HELLO") is True

    def test_whitespace_normalized(self):
        assert exact_match("  hello  world  ", "hello world") is True

    def test_different(self):
        assert exact_match("hello", "world") is False

    def test_none_inputs(self):
        assert exact_match(None, "hello") is False
        assert exact_match("hello", None) is False


class TestCosineSimilarity:
    def test_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_zero_vector(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 2.0])
        assert cosine_similarity(v1, v2) == 0.0


class TestBatchCosineSimilarity:
    def test_basic(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        corpus = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        result = batch_cosine_similarity(query, corpus)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(1.0, abs=1e-5)
        assert result[1] == pytest.approx(0.0, abs=1e-5)
        assert result[2] == pytest.approx(-1.0, abs=1e-5)

    def test_single_vector_corpus(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        corpus = np.array([1.0, 0.0], dtype=np.float32)
        result = batch_cosine_similarity(query, corpus)
        assert result.shape == (1,)

    def test_zero_query(self):
        query = np.zeros(3, dtype=np.float32)
        corpus = np.ones((2, 3), dtype=np.float32)
        result = batch_cosine_similarity(query, corpus)
        assert np.all(result == 0.0)


class TestComputeEmbedding:
    def test_calls_model_encode(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        result = compute_embedding("hello", mock_model)
        mock_model.encode.assert_called_once_with("hello", convert_to_numpy=True)
        assert result.dtype == np.float32


# ======================================================================
# statistics
# ======================================================================

class TestBenfordsExpected:
    def test_first_digit_sums_to_one(self):
        dist = benfords_first_digit_expected()
        assert sum(dist.values()) == pytest.approx(1.0)
        assert set(dist.keys()) == set(range(1, 10))

    def test_first_digit_decreasing(self):
        dist = benfords_first_digit_expected()
        for d in range(1, 9):
            assert dist[d] > dist[d + 1]

    def test_first_two_digits_sums_to_one(self):
        dist = benfords_first_two_digits_expected()
        assert sum(dist.values()) == pytest.approx(1.0)
        assert set(dist.keys()) == set(range(10, 100))


class TestBenfordsTest:
    def test_benford_conforming_data(self):
        # Generate data that roughly follows Benford's distribution
        rng = np.random.default_rng(42)
        amounts = [Decimal(str(round(10 ** rng.uniform(0, 6), 2))) for _ in range(1000)]
        result = benfords_test(amounts, digits=1)
        # Randomly generated power-law-ish data should mostly conform
        assert "chi_squared" in result
        assert "p_value" in result
        assert "mad" in result
        assert "observed_distribution" in result
        assert "expected_distribution" in result

    def test_empty_list(self):
        result = benfords_test([], digits=1)
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_invalid_digits_raises(self):
        with pytest.raises(ValueError):
            benfords_test([Decimal("100")], digits=3)

    def test_uniform_data_detected(self):
        # All amounts start with 5 — highly non-Benford
        amounts = [Decimal(str(500 + i)) for i in range(200)]
        result = benfords_test(amounts, digits=1)
        assert result["significant"] is True

    def test_zeros_ignored(self):
        amounts = [Decimal("0"), Decimal("0"), Decimal("123")]
        result = benfords_test(amounts, digits=1)
        assert result["observed_distribution"].get(1, 0) == 1.0


class TestComputeZscore:
    def test_at_mean(self):
        assert compute_zscore(10.0, 10.0, 2.0) == 0.0

    def test_one_std_above(self):
        assert compute_zscore(12.0, 10.0, 2.0) == pytest.approx(1.0)

    def test_zero_std(self):
        assert compute_zscore(15.0, 10.0, 0.0) == 0.0

    def test_negative_std(self):
        assert compute_zscore(15.0, 10.0, -1.0) == 0.0


class TestPearsonCorrelation:
    def test_perfect_positive(self):
        assert pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert pearson_correlation([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)

    def test_no_correlation(self):
        r = pearson_correlation([1, 2, 3, 4, 5], [5, 1, 4, 2, 3])
        assert abs(r) < 0.5

    def test_short_series(self):
        assert pearson_correlation([1], [2]) == 0.0
        assert pearson_correlation([], []) == 0.0

    def test_zero_variance(self):
        assert pearson_correlation([5, 5, 5], [1, 2, 3]) == 0.0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            pearson_correlation([1, 2], [1, 2, 3])


class TestIsolationForest:
    def test_basic(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 2))
        # Add outliers
        data = np.vstack([data, [[100, 100], [-100, -100]]])
        scores = compute_isolation_forest_scores(data)
        assert len(scores) == 102
        # Outliers should have lower (more negative) scores
        assert scores[-1] < np.median(scores[:100])
        assert scores[-2] < np.median(scores[:100])

    def test_empty(self):
        result = compute_isolation_forest_scores(np.array([]))
        assert len(result) == 0

    def test_single_sample(self):
        result = compute_isolation_forest_scores(np.array([[1.0, 2.0]]))
        assert len(result) == 0

    def test_1d_input(self):
        data = np.array([1.0, 2.0, 3.0, 100.0, 2.5])
        scores = compute_isolation_forest_scores(data)
        assert len(scores) == 5


class TestLinearRegressionTrend:
    def test_perfect_uptrend(self):
        dates = [date(2025, 1, i) for i in range(1, 11)]
        values = [float(i) for i in range(1, 11)]
        result = linear_regression_trend(dates, values)
        assert result["slope"] > 0
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)
        assert result["cumulative_change_pct"] > 0

    def test_flat(self):
        dates = [date(2025, 1, i) for i in range(1, 11)]
        values = [5.0] * 10
        result = linear_regression_trend(dates, values)
        assert result["slope"] == pytest.approx(0.0)

    def test_too_few_points(self):
        result = linear_regression_trend([date(2025, 1, 1)], [5.0])
        assert result["slope"] == 0.0
        assert result["p_value"] == 1.0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            linear_regression_trend(
                [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)],
                [1.0, 2.0],
            )


class TestChiSquaredTest:
    def test_matching_distributions(self):
        obs = {1: 50, 2: 50}
        exp = {1: 50, 2: 50}
        result = chi_squared_test(obs, exp)
        assert result["chi_squared"] == pytest.approx(0.0)
        assert result["significant"] is False

    def test_very_different(self):
        obs = {1: 100, 2: 0}
        exp = {1: 50, 2: 50}
        result = chi_squared_test(obs, exp)
        assert result["chi_squared"] > 0
        assert result["significant"] is True

    def test_empty(self):
        result = chi_squared_test({}, {})
        assert result["p_value"] == 1.0


class TestClusteringBelowThreshold:
    def test_obvious_clustering(self):
        # All amounts just below 5000 threshold
        amounts = [Decimal(str(x)) for x in range(4500, 5000, 10)]
        result = detect_clustering_below_threshold(
            amounts, Decimal("5000"), band_pct=0.10
        )
        assert result["count_in_band"] == len(amounts)
        assert result["significant"] is True

    def test_uniform_distribution(self):
        # Spread amounts evenly from 100 to 4900
        amounts = [Decimal(str(x)) for x in range(100, 5000, 100)]
        result = detect_clustering_below_threshold(
            amounts, Decimal("5000"), band_pct=0.10
        )
        # Should not be highly significant
        assert result["count_in_band"] < len(amounts)

    def test_empty_list(self):
        result = detect_clustering_below_threshold([], Decimal("5000"))
        assert result["count_in_band"] == 0
        assert result["significant"] is False

    def test_zero_threshold(self):
        result = detect_clustering_below_threshold(
            [Decimal("100")], Decimal("0")
        )
        assert result["significant"] is False

    def test_all_above_threshold(self):
        amounts = [Decimal("6000"), Decimal("7000")]
        result = detect_clustering_below_threshold(amounts, Decimal("5000"))
        assert result["count_in_band"] == 0
        assert result["significant"] is False
