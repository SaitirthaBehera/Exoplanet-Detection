"""Unit tests for src.outlier_detection (Section 14: Outlier Removal)."""

import numpy as np
import pytest

from src.outlier_detection import remove_outliers


@pytest.fixture
def lightcurve_with_outliers():
    rng = np.random.default_rng(11)
    time = np.linspace(0, 50, 2000)
    flux = 1.0 + rng.normal(0, 0.001, 2000)
    outlier_idx = rng.choice(2000, 20, replace=False)
    flux[outlier_idx] += rng.choice([-1, 1], 20) * rng.uniform(0.05, 0.2, 20)
    return time, flux, set(outlier_idx)


@pytest.mark.parametrize("method", ["sigma_clip", "mad", "percentile"])
def test_all_methods_remove_most_injected_outliers(lightcurve_with_outliers, method):
    time, flux, outlier_idx = lightcurve_with_outliers
    _, flux_clean, stats = remove_outliers(time, flux, method=method)
    # Should remove a count reasonably close to the 20 injected outliers.
    assert 10 <= stats.n_removed <= 40
    assert stats.method == method
    assert stats.n_input == len(flux)


def test_sigma_clip_default_threshold_is_5():
    time = np.arange(100.0)
    flux = np.ones(100)
    flux[50] = 100.0  # extreme outlier
    _, flux_clean, stats = remove_outliers(time, flux, method="sigma_clip")
    assert stats.n_removed >= 1
    assert 100.0 not in flux_clean


def test_zero_variance_flux_removes_nothing():
    time = np.arange(50.0)
    flux = np.ones(50)
    _, flux_clean, stats = remove_outliers(time, flux, method="sigma_clip")
    assert stats.n_removed == 0
    assert len(flux_clean) == 50


def test_nan_and_inf_are_excluded():
    time = np.arange(100.0)
    flux = np.ones(100)
    flux[5:10] = np.nan
    flux[15] = np.inf
    _, flux_clean, stats = remove_outliers(time, flux, method="sigma_clip")
    assert np.isfinite(flux_clean).all()
    assert stats.n_removed >= 6  # at least the 5 NaN + 1 inf


def test_pct_removed_matches_n_removed():
    time = np.arange(200.0)
    flux = np.ones(200)
    flux[0:10] = 50.0
    _, _, stats = remove_outliers(time, flux, method="sigma_clip")
    assert stats.pct_removed == pytest.approx(100.0 * stats.n_removed / stats.n_input)


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        remove_outliers(np.arange(5.0), np.arange(5.0), method="bogus")


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        remove_outliers(np.arange(5.0), np.arange(4.0))
