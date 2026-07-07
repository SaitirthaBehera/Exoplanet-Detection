"""
Training-only data augmentation for light curves (Section 8).

Provides five configurable augmentation techniques, each applied
independently with its own probability:

* **Gaussian noise** — add random noise, simulating instrument/photon noise.
* **Time shift** — roll the flux array by a small number of samples,
  simulating imprecise transit-phase alignment.
* **Flux scaling** — multiply by a scalar near 1.0, simulating calibration
  uncertainty.
* **Random masking** — zero out (or NaN-mask) a short contiguous span,
  simulating data gaps / bad-quality-flag removal.
* **Baseline drift** — add a slow sinusoidal trend, simulating residual
  stellar variability that detrending didn't fully remove.

Per Section 8, augmentation must never run at inference time — the
:func:`augment` orchestrator function takes an explicit ``training`` flag
and is a no-op when ``training=False``, and each individual augmentation
function is also usable standalone for unit testing / manual pipelines.

Example
-------
>>> from src.augmentation import augment
>>> flux_aug = augment(flux, config=aug_config, training=True)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def add_gaussian_noise(
    flux: np.ndarray,
    sigma_range: Tuple[float, float] = (0.0005, 0.003),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add Gaussian noise with a randomly chosen sigma from ``sigma_range``.

    This mirrors ``src.preprocessing.augment_with_noise`` exactly (same
    default range, same sampling scheme), just exposed with an injectable
    RNG for reproducibility.
    """
    rng = rng or np.random.default_rng()
    sigma = rng.uniform(*sigma_range)
    noise = rng.normal(0.0, sigma, size=flux.shape)
    logger.debug("Applied Gaussian noise augmentation (sigma=%.5f).", sigma)
    return flux + noise


def apply_time_shift(
    flux: np.ndarray,
    max_shift_fraction: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Circularly shift the flux array by a small random number of samples.

    Parameters
    ----------
    flux:
        1-D flux array.
    max_shift_fraction:
        Maximum shift, as a fraction of ``len(flux)``, in either direction.
    rng:
        Optional random generator for reproducibility.
    """
    rng = rng or np.random.default_rng()
    max_shift = max(int(len(flux) * max_shift_fraction), 1)
    shift = int(rng.integers(-max_shift, max_shift + 1))
    logger.debug("Applied time-shift augmentation (shift=%d samples).", shift)
    return np.roll(flux, shift)


def apply_flux_scaling(
    flux: np.ndarray,
    scale_range: Tuple[float, float] = (0.98, 1.02),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Multiply flux by a random scalar drawn from ``scale_range``."""
    rng = rng or np.random.default_rng()
    scale = rng.uniform(*scale_range)
    logger.debug("Applied flux-scaling augmentation (scale=%.4f).", scale)
    return flux * scale


def apply_random_masking(
    flux: np.ndarray,
    max_mask_fraction: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Zero out a short contiguous span of the flux array.

    Parameters
    ----------
    flux:
        1-D flux array.
    max_mask_fraction:
        Maximum length of the masked span, as a fraction of ``len(flux)``.
    rng:
        Optional random generator for reproducibility.
    """
    rng = rng or np.random.default_rng()
    n = len(flux)
    max_mask_len = max(int(n * max_mask_fraction), 1)
    mask_len = int(rng.integers(1, max_mask_len + 1))
    start = int(rng.integers(0, max(n - mask_len, 1) + 1))
    start = min(start, n - mask_len) if n >= mask_len else 0

    flux_out = flux.copy()
    flux_out[start:start + mask_len] = 0.0
    logger.debug(
        "Applied random-masking augmentation (span=[%d:%d], len=%d).",
        start, start + mask_len, mask_len,
    )
    return flux_out


def apply_baseline_drift(
    flux: np.ndarray,
    amplitude_range: Tuple[float, float] = (0.0002, 0.002),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add a slow sinusoidal baseline drift, simulating residual stellar variability.

    Uses the same style of drift as
    ``src.preprocessing.generate_synthetic_dataset``'s synthetic trend
    (1-3 cycles, random phase) so augmented real data resembles the
    distribution the models were trained on.
    """
    rng = rng or np.random.default_rng()
    n = len(flux)
    amplitude = rng.uniform(*amplitude_range)
    n_cycles = rng.uniform(1.0, 3.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    drift = amplitude * np.sin(2.0 * np.pi * n_cycles * np.arange(n) / n + phase)
    logger.debug(
        "Applied baseline-drift augmentation (amplitude=%.5f, cycles=%.2f).",
        amplitude, n_cycles,
    )
    return flux + drift


def augment(
    flux: np.ndarray,
    config: Optional["AugmentationConfig"] = None,  # noqa: F821 (type-check only import)
    training: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply the full configurable augmentation pipeline to a flux array.

    Each of the five augmentation techniques (Gaussian noise, time shift,
    flux scaling, random masking, baseline drift) is applied independently
    with its own configured probability, in that fixed order.

    Parameters
    ----------
    flux:
        1-D flux array to augment.
    config:
        An :class:`src.preprocessing_config.AugmentationConfig`. If
        ``None``, augmentation defaults are loaded from
        :func:`src.preprocessing_config.load_preprocessing_config`.
    training:
        If ``False``, this function is a strict no-op and returns
        ``flux`` unchanged (a copy). **Augmentation must never be applied
        at inference time** — callers should always pass
        ``training=False`` (or simply not call this function) outside of
        the training data pipeline.
    rng:
        Optional random generator for reproducible augmentation.

    Returns
    -------
    np.ndarray
        The augmented flux array (or an unmodified copy if
        ``training=False`` or ``config.enabled=False``).
    """
    flux = np.asarray(flux, dtype=np.float64)

    if not training:
        logger.debug("augment() called with training=False; returning flux unchanged.")
        return flux.copy()

    if config is None:
        from src.preprocessing_config import load_preprocessing_config
        config = load_preprocessing_config().augmentation

    if not config.enabled:
        logger.debug("Augmentation disabled in config; returning flux unchanged.")
        return flux.copy()

    rng = rng or np.random.default_rng()
    out = flux.copy()
    applied = []

    if rng.random() < config.gaussian_noise.probability:
        out = add_gaussian_noise(
            out,
            sigma_range=config.gaussian_noise.params.get("sigma_range", (0.0005, 0.003)),
            rng=rng,
        )
        applied.append("gaussian_noise")

    if rng.random() < config.time_shift.probability:
        out = apply_time_shift(
            out,
            max_shift_fraction=config.time_shift.params.get("max_shift_fraction", 0.02),
            rng=rng,
        )
        applied.append("time_shift")

    if rng.random() < config.flux_scaling.probability:
        out = apply_flux_scaling(
            out,
            scale_range=config.flux_scaling.params.get("scale_range", (0.98, 1.02)),
            rng=rng,
        )
        applied.append("flux_scaling")

    if rng.random() < config.random_masking.probability:
        out = apply_random_masking(
            out,
            max_mask_fraction=config.random_masking.params.get("max_mask_fraction", 0.05),
            rng=rng,
        )
        applied.append("random_masking")

    if rng.random() < config.baseline_drift.probability:
        out = apply_baseline_drift(
            out,
            amplitude_range=config.baseline_drift.params.get("amplitude_range", (0.0002, 0.002)),
            rng=rng,
        )
        applied.append("baseline_drift")

    logger.info("Augmentation applied: %s", applied if applied else "none (all probability rolls missed)")

    return out
