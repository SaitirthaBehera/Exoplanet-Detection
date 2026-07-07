"""Unit tests for src.feature_statistics (Section 14: Statistics)."""

import numpy as np
import pytest

from src.feature_statistics import compute_statistics
from src.data_validation import validate_lightcurve


def test_basic_statistics_match_numpy(clean_lightcurve):
    time, flux = clean_lightcurve
    stats = compute_statistics(time, flux)
    assert stats.mean_flux == pytest.approx(np.mean(flux))
    assert stats.median_flux == pytest.approx(np.median(flux))
    assert stats.std_flux == pytest.approx(np.std(flux))
    assert stats.variance_flux == pytest.approx(np.var(flux))
    assert stats.min_flux == pytest.approx(np.min(flux))
    assert stats.max_flux == pytest.approx(np.max(flux))
    assert stats.rms == pytest.approx(np.sqrt(np.mean(flux ** 2)))
    assert stats.n_samples == len(flux)


def test_missing_value_count(messy_lightcurve):
    time, flux = messy_lightcurve
    stats = compute_statistics(time, flux)
    assert stats.missing_value_count == int(np.sum(np.isnan(flux)))


def test_gap_count_detects_injected_gap(messy_lightcurve):
    time, flux = messy_lightcurve
    stats = compute_statistics(time, flux)
    assert stats.gap_count >= 1


def test_all_nan_returns_zeroed_stats():
    stats = compute_statistics(np.arange(5.0), np.full(5, np.nan))
    assert stats.mean_flux == 0.0
    assert stats.n_samples == 5
    assert stats.missing_value_count == 5


def test_noise_estimate_consistent_with_data_validation(clean_lightcurve):
    time, flux = clean_lightcurve
    stats = compute_statistics(time, flux)
    report = validate_lightcurve(time, flux)
    assert stats.estimated_noise == pytest.approx(report.estimated_noise)


def test_to_dict_is_json_serializable(clean_lightcurve):
    import json
    time, flux = clean_lightcurve
    stats = compute_statistics(time, flux)
    json.dumps(stats.to_dict())  # should not raise


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compute_statistics(np.arange(5.0), np.arange(4.0))
