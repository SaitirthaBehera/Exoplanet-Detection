"""Unit tests for src.data_validation (Section 2: Data Validation)."""

import numpy as np

from src.data_validation import validate_lightcurve


def test_clean_lightcurve_is_valid(clean_lightcurve):
    time, flux = clean_lightcurve
    report = validate_lightcurve(time, flux, target_id="Clean-1", mission="Kepler")
    assert report.is_valid
    assert report.n_observations == len(flux)
    assert report.missing_value_pct == 0.0
    assert report.n_duplicate_timestamps == 0
    assert report.n_infinite_values == 0
    assert len(report.warnings) == 0


def test_messy_lightcurve_flags_all_issues(messy_lightcurve):
    time, flux = messy_lightcurve
    report = validate_lightcurve(time, flux, target_id="Messy-1", mission="TESS")
    assert report.is_valid  # quality issues alone should not invalidate
    assert report.missing_value_pct > 0
    assert report.n_duplicate_timestamps >= 1
    assert len(report.warnings) >= 3


def test_empty_lightcurve_is_invalid():
    report = validate_lightcurve(np.array([]), np.array([]))
    assert report.is_valid is False
    assert report.n_observations == 0
    assert report.missing_value_pct == 100.0


def test_all_nan_flux_is_invalid():
    report = validate_lightcurve(np.arange(10.0), np.full(10, np.nan))
    assert report.is_valid is False


def test_noise_estimate_is_reasonable(clean_lightcurve):
    time, flux = clean_lightcurve
    report = validate_lightcurve(time, flux)
    # True injected noise sigma is 0.001; estimator should be in the right ballpark.
    assert 0.0005 < report.estimated_noise < 0.002


def test_report_to_dict_is_json_serializable(clean_lightcurve):
    import json
    time, flux = clean_lightcurve
    report = validate_lightcurve(time, flux, target_id="X", mission="Kepler")
    d = report.to_dict()
    json.dumps(d)  # should not raise
    assert d["target_id"] == "X"
    assert isinstance(d["time_range"], list)


def test_negative_flux_is_flagged():
    time = np.arange(20.0)
    flux = np.ones(20)
    flux[5] = -1.0
    report = validate_lightcurve(time, flux)
    assert report.n_negative_flux == 1
    assert any("negative" in w.lower() for w in report.warnings)
