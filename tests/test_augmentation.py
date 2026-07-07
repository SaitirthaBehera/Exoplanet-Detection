"""Unit tests for src.augmentation (Section 8: Data Augmentation)."""

import numpy as np
import pytest

from src.augmentation import (
    add_gaussian_noise,
    apply_baseline_drift,
    apply_flux_scaling,
    apply_random_masking,
    apply_time_shift,
    augment,
)
from src.preprocessing_config import load_preprocessing_config


@pytest.fixture
def flat_flux():
    rng = np.random.default_rng(5)
    return np.ones(1000) + rng.normal(0, 0.001, 1000)


def test_gaussian_noise_changes_values(flat_flux):
    out = add_gaussian_noise(flat_flux, rng=np.random.default_rng(1))
    assert not np.array_equal(out, flat_flux)
    assert out.shape == flat_flux.shape


def test_time_shift_reorders_values(flat_flux):
    flux = np.arange(100.0)
    # max_shift_fraction=0.02 on 100 samples gives max_shift=2, so shift is
    # drawn from {-2,-1,0,1,2}; try a few seeds to avoid the (rare) shift=0 case.
    shifted_at_least_once = False
    for seed in range(10):
        out = apply_time_shift(flux, rng=np.random.default_rng(seed))
        if not np.array_equal(out, flux):
            shifted_at_least_once = True
            # A roll should preserve the same multiset of values.
            np.testing.assert_array_equal(np.sort(out), np.sort(flux))
    assert shifted_at_least_once


def test_flux_scaling_stays_near_one(flat_flux):
    out = apply_flux_scaling(flat_flux, scale_range=(0.98, 1.02), rng=np.random.default_rng(1))
    ratio = out / flat_flux
    assert 0.97 < ratio.mean() < 1.03


def test_random_masking_zeros_a_span(flat_flux):
    out = apply_random_masking(flat_flux, max_mask_fraction=0.1, rng=np.random.default_rng(1))
    assert np.sum(out == 0.0) > 0


def test_baseline_drift_adds_variation(flat_flux):
    out = apply_baseline_drift(flat_flux, rng=np.random.default_rng(1))
    assert not np.array_equal(out, flat_flux)


def test_augment_training_false_is_exact_noop(flat_flux):
    cfg = load_preprocessing_config().augmentation
    cfg.enabled = True
    for step in [cfg.gaussian_noise, cfg.time_shift, cfg.flux_scaling,
                 cfg.random_masking, cfg.baseline_drift]:
        step.probability = 1.0
    out = augment(flat_flux, config=cfg, training=False)
    np.testing.assert_array_equal(out, flat_flux)


def test_augment_disabled_config_is_noop(flat_flux):
    cfg = load_preprocessing_config().augmentation
    cfg.enabled = False
    out = augment(flat_flux, config=cfg, training=True)
    np.testing.assert_array_equal(out, flat_flux)


def test_augment_zero_probability_is_noop(flat_flux):
    cfg = load_preprocessing_config().augmentation
    cfg.enabled = True
    for step in [cfg.gaussian_noise, cfg.time_shift, cfg.flux_scaling,
                 cfg.random_masking, cfg.baseline_drift]:
        step.probability = 0.0
    out = augment(flat_flux, config=cfg, training=True, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(out, flat_flux)


def test_augment_full_probability_changes_flux(flat_flux):
    cfg = load_preprocessing_config().augmentation
    cfg.enabled = True
    for step in [cfg.gaussian_noise, cfg.time_shift, cfg.flux_scaling,
                 cfg.random_masking, cfg.baseline_drift]:
        step.probability = 1.0
    out = augment(flat_flux, config=cfg, training=True, rng=np.random.default_rng(7))
    assert not np.array_equal(out, flat_flux)


def test_augment_is_reproducible_with_seeded_rng(flat_flux):
    cfg = load_preprocessing_config().augmentation
    cfg.enabled = True
    cfg.gaussian_noise.probability = 1.0
    out1 = augment(flat_flux, config=cfg, training=True, rng=np.random.default_rng(99))
    out2 = augment(flat_flux, config=cfg, training=True, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(out1, out2)
