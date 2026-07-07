"""Tests for src.preprocessing_pipeline (Sections 11-13: Orchestration, Outputs, Logging)."""

import json
import os

import numpy as np
import pytest

from src.preprocessing_config import load_preprocessing_config
from src.preprocessing_pipeline import PreprocessingPipeline


EXPECTED_STAGE_NAMES = [
    "load_data", "validate", "handle_missing_values", "remove_outliers",
    "normalize", "detrend", "window_generation", "statistics",
    "augmentation", "save_output",
]


def test_pipeline_runs_all_stages_in_order(messy_lightcurve, tmp_path):
    time, flux = messy_lightcurve
    pipeline = PreprocessingPipeline()
    result = pipeline.run(
        time, flux, target_id="Messy-1", mission="Kepler",
        save_output=True, output_dir=str(tmp_path / "Messy-1"),
    )
    stage_names = [s.stage_name for s in result.stage_logs]
    assert stage_names == EXPECTED_STAGE_NAMES


def test_pipeline_produces_valid_final_flux(clean_lightcurve, tmp_path):
    time, flux = clean_lightcurve
    pipeline = PreprocessingPipeline()
    result = pipeline.run(
        time, flux, target_id="Clean-1", save_output=False,
    )
    assert np.isfinite(result.flux).all()
    assert len(result.windows) > 0


def test_pipeline_saves_all_required_output_files(clean_lightcurve, tmp_path):
    time, flux = clean_lightcurve
    out_dir = tmp_path / "TestTarget"
    pipeline = PreprocessingPipeline()
    result = pipeline.run(
        time, flux, target_id="TestTarget", save_output=True, output_dir=str(out_dir),
    )
    assert result.output_dir == str(out_dir)
    assert (out_dir / "processed.csv").exists()
    assert (out_dir / "statistics.json").exists()
    assert (out_dir / "preprocessing_report.json").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "windows" / "windows.npz").exists()

    with open(out_dir / "preprocessing_report.json") as f:
        report = json.load(f)
    assert report["target_id"] == "TestTarget"
    assert len(report["stages"]) == len(EXPECTED_STAGE_NAMES)


def test_pipeline_save_output_false_skips_files(clean_lightcurve, tmp_path):
    time, flux = clean_lightcurve
    pipeline = PreprocessingPipeline()
    result = pipeline.run(time, flux, target_id="NoSave", save_output=False)
    assert result.output_dir is None


def test_pipeline_is_reproducible_with_same_seed(clean_lightcurve):
    time, flux = clean_lightcurve
    cfg = load_preprocessing_config()
    cfg.augmentation.enabled = True
    cfg.augmentation.gaussian_noise.probability = 1.0

    p1 = PreprocessingPipeline(config=cfg)
    r1 = p1.run(time, flux, target_id="Repro", apply_augmentation=True, save_output=False)

    p2 = PreprocessingPipeline(config=cfg)
    r2 = p2.run(time, flux, target_id="Repro", apply_augmentation=True, save_output=False)

    np.testing.assert_array_equal(r1.flux, r2.flux)


def test_pipeline_error_is_logged_and_reraised():
    pipeline = PreprocessingPipeline()
    with pytest.raises(ValueError):
        pipeline.run(np.arange(10.0), np.arange(5.0), target_id="Bad", save_output=False)
    # Stages that ran before the failure should still be recorded.
    stage_names = [s.stage_name for s in pipeline.stage_logs]
    assert "load_data" in stage_names
    assert pipeline.stage_logs[-1].errors  # last stage recorded the error


def test_pipeline_detrend_disabled_by_default_is_noop(clean_lightcurve):
    time, flux = clean_lightcurve
    pipeline = PreprocessingPipeline()  # default config has detrending disabled
    result = pipeline.run(time, flux, target_id="NoDetrend", save_output=False)
    detrend_log = next(s for s in result.stage_logs if s.stage_name == "detrend")
    assert detrend_log.config_used.get("enabled") is False


def test_run_from_source_with_array_tuple(clean_lightcurve):
    time, flux = clean_lightcurve
    pipeline = PreprocessingPipeline()
    result = pipeline.run_from_source(
        (time, flux), target_id="FromArray", save_output=False,
    )
    assert result.target_id == "FromArray"
    assert len(result.flux) > 0


def test_run_from_source_with_csv(tmp_path, clean_lightcurve):
    import pandas as pd
    time, flux = clean_lightcurve
    csv_path = tmp_path / "lc.csv"
    pd.DataFrame({"time": time, "flux": flux}).to_csv(csv_path, index=False)

    pipeline = PreprocessingPipeline()
    result = pipeline.run_from_source(
        str(csv_path), target_id="FromCSV", mission="Kepler", save_output=False,
    )
    assert result.target_id == "FromCSV"
    assert result.mission == "Kepler"
