"""Shared pytest fixtures for the ExoVision AI preprocessing test suite."""

import numpy as np
import pytest


@pytest.fixture
def clean_lightcurve():
    """A clean, noise-free synthetic light curve (uniform cadence, no gaps)."""
    rng = np.random.default_rng(42)
    time = np.linspace(0, 50, 5000)
    flux = 1.0 + rng.normal(0, 0.001, 5000)
    return time, flux


@pytest.fixture
def messy_lightcurve():
    """A light curve with NaNs, an outlier, a duplicate timestamp, and a gap."""
    rng = np.random.default_rng(7)
    time = np.linspace(0, 50, 2000)
    flux = 1.0 + rng.normal(0, 0.001, 2000)
    flux[50:60] = np.nan
    flux[500] += 0.2  # outlier
    time[900] = time[899]  # duplicate timestamp
    time[1500:1550] += 5.0  # gap
    return time, flux


@pytest.fixture
def transit_lightcurve():
    """A light curve with slow stellar variability and a short transit dip."""
    rng = np.random.default_rng(3)
    n = 3000
    time = np.linspace(0, 60, n)
    trend = 1.0 + 0.01 * np.sin(2 * np.pi * time / 40)
    flux = trend.copy()
    flux[1480:1520] -= 0.01  # short transit-like dip
    flux += rng.normal(0, 0.0005, n)
    return time, flux
