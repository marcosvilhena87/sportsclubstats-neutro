import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add the src directory to the module search path so we can import simulator
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import simulator  # type: ignore


def _naive_poisson_ppf(u: float, lam: float) -> int:
    k = 0
    p = math.exp(-lam)
    cdf = p
    while u > cdf:
        k += 1
        p *= lam / k
        cdf += p
    return k


def test_poisson_ppf_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    for lam in [1, 5, 20, 1000]:
        us = np.linspace(0.01, 0.99, 5)
        for u in us:
            assert simulator._poisson_ppf(u, lam) == int(scipy_stats.poisson.ppf(u, lam))


def test_poisson_ppf_uses_scipy(monkeypatch):
    scipy_stats = pytest.importorskip("scipy.stats")
    calls = {"count": 0}

    original = scipy_stats.poisson.ppf

    def fake_ppf(u, lam):
        calls["count"] += 1
        return original(u, lam)

    monkeypatch.setattr(scipy_stats.poisson, "ppf", fake_ppf)
    simulator._poisson_ppf(0.5, 5.0)
    assert calls["count"] == 1


def test_poisson_ppf_fallback_speed(monkeypatch):
    """Optimised fallback should beat the old cumulative loop."""

    monkeypatch.setattr(simulator, "_scipy_poisson", None)
    simulator._poisson_cdf_array.cache_clear()

    lam = 50.0
    us = np.random.default_rng(0).random(1000)

    start = time.time()
    for u in us:
        simulator._poisson_ppf(u, lam)
    new_time = time.time() - start

    start = time.time()
    for u in us:
        _naive_poisson_ppf(u, lam)
    old_time = time.time() - start

    assert new_time < old_time
