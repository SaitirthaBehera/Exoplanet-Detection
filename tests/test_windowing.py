"""Unit tests for src.windowing (Section 14: Window Generation)."""

import numpy as np
import pytest

from src.windowing import Window, generate_windows, windows_to_arrays
from src.preprocessing import create_windows


def test_matches_legacy_create_windows_exactly(clean_lightcurve):
    time, flux = clean_lightcurve
    time = np.linspace(0, 100, 8000)
    rng = np.random.default_rng(4)
    flux = np.sin(time / 5) + rng.normal(0, 0.01, 8000)

    flux_legacy, time_legacy = create_windows(time, flux, window_size=2001, stride=500)

    overlap = 1 - 500 / 2001
    windows = generate_windows(time, flux, window_size=2001, overlap=overlap)
    arrs = windows_to_arrays(windows)

    assert len(windows) == len(flux_legacy)
    np.testing.assert_allclose(arrs["flux"], flux_legacy)
    np.testing.assert_allclose(arrs["time"], time_legacy)


def test_window_ids_are_unique_and_ordered():
    time = np.linspace(0, 10, 3000)
    flux = np.ones(3000)
    windows = generate_windows(time, flux, window_size=500, overlap=0.5)
    ids = [w.window_id for w in windows]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_each_window_has_correct_size():
    time = np.linspace(0, 10, 3000)
    flux = np.ones(3000)
    windows = generate_windows(time, flux, window_size=500, overlap=0.5)
    for w in windows:
        assert w.flux.shape == (500,)
        assert w.time.shape == (500,)


def test_short_lightcurve_gets_padded():
    time = np.arange(500.0)
    flux = np.ones(500)
    windows = generate_windows(time, flux, window_size=2001, overlap=0.75)
    assert len(windows) == 1
    assert windows[0].flux.shape == (2001,)
    assert windows[0].metadata["n_padded"] > 0


def test_extra_metadata_propagates():
    time = np.linspace(0, 10, 1000)
    flux = np.ones(1000)
    windows = generate_windows(
        time, flux, window_size=200, overlap=0.5,
        extra_metadata={"target_id": "Kepler-10b", "mission": "Kepler"},
    )
    assert all(w.metadata["target_id"] == "Kepler-10b" for w in windows)
    assert all(w.metadata["mission"] == "Kepler" for w in windows)


def test_windows_to_arrays_empty_list():
    arrs = windows_to_arrays([])
    assert arrs["flux"].shape == (0, 0)
    assert arrs["window_ids"].shape == (0,)


@pytest.mark.parametrize("bad_kwargs,match", [
    (dict(window_size=0), "window_size"),
    (dict(overlap=1.0), "overlap"),
])
def test_invalid_params_raise(bad_kwargs, match):
    time = np.arange(100.0)
    flux = np.ones(100)
    with pytest.raises(ValueError, match=match):
        generate_windows(time, flux, **bad_kwargs)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        generate_windows(np.arange(10.0), np.arange(5.0))
