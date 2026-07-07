"""Unit tests for src.normalization (Section 14: Normalization)."""

import numpy as np
import pytest

from src.normalization import normalize
from src.preprocessing import normalize_flux


def test_median_method_matches_legacy_normalize_flux(clean_lightcurve):
    _, flux = clean_lightcurve
    new_out, params = normalize(flux, method="median")
    legacy_out = normalize_flux(flux)
    np.testing.assert_allclose(new_out, legacy_out)
    assert params["method"] == "median"


def test_minmax_output_range():
    flux = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out, params = normalize(flux, method="minmax")
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    assert params["min"] == 1.0
    assert params["max"] == 5.0


def test_zscore_has_zero_mean_unit_std():
    rng = np.random.default_rng(1)
    flux = rng.normal(10, 2, 1000)
    out, params = normalize(flux, method="zscore")
    assert out.mean() == pytest.approx(0.0, abs=1e-9)
    assert out.std() == pytest.approx(1.0, abs=1e-9)


def test_robust_uses_median_and_iqr():
    flux = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # 100 is an outlier
    out, params = normalize(flux, method="robust")
    assert params["method"] == "robust"
    assert "iqr" in params
    # Robust scaling should be far less distorted by the outlier than minmax would be.
    assert out[-1] > out[0]


@pytest.mark.parametrize("method", ["minmax", "zscore", "median", "robust"])
def test_constant_flux_does_not_raise(method):
    flux = np.ones(50)
    out, params = normalize(flux, method=method)
    assert len(out) == 50
    assert np.isfinite(out).all()


def test_zero_median_does_not_divide_by_zero():
    flux = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    out, params = normalize(flux, method="median")
    assert np.isfinite(out).all()


def test_empty_flux_returns_empty():
    out, params = normalize(np.array([]), method="median")
    assert len(out) == 0


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        normalize(np.arange(5.0), method="bogus")
