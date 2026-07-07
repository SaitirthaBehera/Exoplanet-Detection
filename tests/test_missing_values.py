"""Unit tests for src.missing_values (Section 14: Interpolation)."""

import numpy as np
import pytest

from src.missing_values import handle_missing_values


@pytest.mark.parametrize("strategy", ["linear", "cubic", "nearest", "ffill", "bfill"])
def test_all_strategies_fill_gaps(strategy):
    time = np.arange(20.0)
    flux = np.sin(time / 3) + 1.0
    flux_gapped = flux.copy()
    flux_gapped[[3, 4, 5, 10, 15]] = np.nan

    filled, n_filled = handle_missing_values(time, flux_gapped, strategy=strategy)

    assert n_filled == 5
    assert not np.isnan(filled).any()


def test_linear_interpolation_is_accurate_on_linear_signal():
    time = np.arange(30.0)
    flux = 2.0 * time + 5.0  # perfectly linear
    flux_gapped = flux.copy()
    flux_gapped[10:15] = np.nan

    filled, n_filled = handle_missing_values(time, flux_gapped, strategy="linear")

    assert n_filled == 5
    np.testing.assert_allclose(filled, flux, atol=1e-9)


def test_no_missing_values_returns_unchanged():
    time = np.arange(10.0)
    flux = np.ones(10)
    filled, n_filled = handle_missing_values(time, flux, strategy="linear")
    assert n_filled == 0
    np.testing.assert_array_equal(filled, flux)


def test_all_missing_values_returns_unfilled():
    time = np.arange(10.0)
    flux = np.full(10, np.nan)
    filled, n_filled = handle_missing_values(time, flux, strategy="linear")
    assert n_filled == 0
    assert np.isnan(filled).all()


def test_cubic_falls_back_to_linear_with_few_points():
    time = np.arange(3.0)
    flux = np.array([1.0, np.nan, 3.0])
    filled, n_filled = handle_missing_values(time, flux, strategy="cubic")
    assert n_filled == 1
    assert filled[1] == pytest.approx(2.0)


def test_ffill_propagates_forward():
    time = np.arange(5.0)
    flux = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
    filled, _ = handle_missing_values(time, flux, strategy="ffill")
    assert filled[1] == 1.0
    assert filled[2] == 1.0
    assert filled[4] == 4.0  # trailing NaN with no later value backfills from last valid


def test_bfill_propagates_backward():
    time = np.arange(5.0)
    flux = np.array([np.nan, 2.0, np.nan, np.nan, 5.0])
    filled, _ = handle_missing_values(time, flux, strategy="bfill")
    assert filled[0] == 2.0  # leading NaN forward-filled from first valid
    assert filled[2] == 5.0
    assert filled[3] == 5.0


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        handle_missing_values(np.arange(5.0), np.arange(5.0), strategy="bogus")


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        handle_missing_values(np.arange(5.0), np.arange(4.0))
