"""Unit tests for src.input_adapters (Section 1: Supported Inputs)."""

import numpy as np
import pandas as pd
import pytest

from src.input_adapters import detect_input_type, load_input


@pytest.fixture
def sample_arrays():
    time = np.linspace(0, 10, 100)
    flux = 1.0 + 0.001 * np.sin(time)
    return time, flux


def test_detect_and_load_ndarray_pair(sample_arrays):
    time, flux = sample_arrays
    assert detect_input_type((time, flux)) == "ndarray_pair"
    t, f, meta = load_input((time, flux))
    np.testing.assert_allclose(t, time)
    assert meta["source_type"] == "ndarray"


def test_detect_and_load_ndarray_2d(sample_arrays):
    time, flux = sample_arrays
    arr = np.column_stack([time, flux])
    assert detect_input_type(arr) == "ndarray_2d"
    t, f, meta = load_input(arr)
    np.testing.assert_allclose(t, time)


def test_detect_and_load_dataframe_standard_columns(sample_arrays):
    time, flux = sample_arrays
    df = pd.DataFrame({"time": time, "flux": flux})
    assert detect_input_type(df) == "dataframe"
    t, f, meta = load_input(df)
    assert meta["time_column"] == "time"
    assert meta["flux_column"] == "flux"


def test_dataframe_auto_detects_alternate_column_names(sample_arrays):
    time, flux = sample_arrays
    df = pd.DataFrame({"BJD": time, "PDCSAP_FLUX": flux})
    t, f, meta = load_input(df)
    assert meta["time_column"] == "BJD"
    assert meta["flux_column"] == "PDCSAP_FLUX"


def test_dataframe_unrecognized_columns_requires_explicit(sample_arrays):
    time, flux = sample_arrays
    df = pd.DataFrame({"x": time, "y": flux})
    with pytest.raises(ValueError):
        load_input(df)
    t, f, meta = load_input(df, time_col="x", flux_col="y")
    np.testing.assert_allclose(t, time)


def test_csv_round_trip(tmp_path, sample_arrays):
    time, flux = sample_arrays
    df = pd.DataFrame({"time": time, "flux": flux})
    csv_path = tmp_path / "lc.csv"
    df.to_csv(csv_path, index=False)

    assert detect_input_type(str(csv_path)) == "csv"
    t, f, meta = load_input(str(csv_path))
    np.testing.assert_allclose(t, time)
    assert meta["source_type"] == "csv"


def test_bad_file_extension_raises():
    with pytest.raises(ValueError):
        load_input("/tmp/some_file.txt")


def test_mismatched_length_tuple_raises(sample_arrays):
    time, flux = sample_arrays
    with pytest.raises(ValueError):
        load_input((time, flux[:10]))


def test_lightkurve_like_object_duck_typing(sample_arrays):
    time, flux = sample_arrays

    class FakeQuantity:
        def __init__(self, value):
            self.value = value

    class FakeLightCurve:
        def __init__(self, time, flux, targetid=None, mission=None):
            self.time = FakeQuantity(time)
            self.flux = FakeQuantity(flux)
            self.targetid = targetid
            self.mission = mission

    lc = FakeLightCurve(time, flux, targetid=11904151, mission="Kepler")
    assert detect_input_type(lc) == "lightkurve"
    t, f, meta = load_input(lc)
    np.testing.assert_allclose(t, time)
    assert meta["target_id"] == 11904151
    assert meta["mission"] == "Kepler"


def test_unrecognized_type_raises():
    with pytest.raises(ValueError):
        load_input(42)


def test_bad_shape_ndarray_raises():
    with pytest.raises(ValueError):
        load_input(np.zeros((5, 3)))
