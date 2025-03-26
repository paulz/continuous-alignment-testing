from math import sqrt
from statistics import NormalDist

import pytest
from helpers import is_within_expected


def test_binomial_approx():
    # https://docs.python.org/3/library/statistics.html#approximating-binomial-distributions
    n = 750
    p = 0.65
    q = 1 - p
    k = 500
    dist = NormalDist(mu=n * p, sigma=sqrt(n * p * q)).cdf(k + 0.5)
    assert dist == pytest.approx(0.84, rel=0.001)


def test_compare_with_statistical_analysis():
    n = 750
    p = 0.65
    k = 500
    assert is_within_expected(success_rate=p, failure_count=n - k, sample_size=n)
