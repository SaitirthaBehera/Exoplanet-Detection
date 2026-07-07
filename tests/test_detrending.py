"""Unit tests for src.detrending (Section 6: Trend Removal)."""

import numpy as np
import pytest
from scipy.signal import savgol_filter

from src.detrending import detrend


@pytest.mark.parametrize("method,kwargs", [
    ("polynomial", dict(polynomial_degree=3)),
    ("savgol", dict(savgol_window=401, savgol_polyorder=3)),
    ("running_median", dict(running_median_window=201)),
])
def test_all_methods_preserve_transit_dip(transit_lightcurve, method, kwargs):
    time, flux = transit_lightcurve
    flux_flat, trend = detrend(flux, method=method, **kwargs)
    dip_depth_recovered = 1.0 - flux_flat[1500]
    # True injected dip depth is 0.01; recovered depth should be in the right ballpark.
    assert 0.003 < dip_depth_recovered < 0.02


def test_savgol_matches_flatten_lightcurve_convention(transit_lightcurve):
    _, flux = transit_lightcurve
    window_length = 401
    trend_ref = savgol_filter(flux, window_length=window_length, polyorder=3)
    trend_ref_safe = np.where(np.abs(trend_ref) < 1e-10, 1.0, trend_ref)
    flat_ref = flux / trend_ref_safe

    flat_new, _ = detrend(flux, method="savgol", savgol_window=401, savgol_polyorder=3)
    np.testing.assert_allclose(flat_ref, flat_new)


def test_nan_input_does_not_propagate_to_trend(transit_lightcurve):
    _, flux = transit_lightcurve
    flux_with_nan = flux.copy()
    flux_with_nan[100:110] = np.nan
    _, trend = detrend(flux_with_nan, method="running_median", running_median_window=101)
    assert not np.isnan(trend).any()


def test_short_array_falls_back_gracefully():
    flux = np.array([1.0, 1.01, 0.99, 1.0, 1.02])
    flux_flat, trend = detrend(flux, method="savgol", savgol_window=401, savgol_polyorder=3)
    assert len(flux_flat) == 5
    assert np.isfinite(flux_flat).all()


def test_empty_array_does_not_raise():
    flux_flat, trend = detrend(np.array([]))
    assert len(flux_flat) == 0


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        detrend(np.arange(10.0) + 1.0, method="bogus")
