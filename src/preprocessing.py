import logging
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


def normalize_flux(flux: np.ndarray) -> np.ndarray:
    median = np.median(flux)
    if np.abs(median) < 1e-15:
        logger.warning("Median flux is near zero (%.3e); skipping division.",
                        median)
        return flux
    normalized = (flux / median) - 1.0
    logger.info("Normalized flux: median=%.6f → baseline ≈ 0.", median)
    return normalized


def create_windows(
    time: np.ndarray,
    flux: np.ndarray,
    window_size: int = 2001,
    stride: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(flux)
    if n < window_size:
        pad_len = window_size - n
        flux = np.pad(flux, (0, pad_len), mode="edge")
        time = np.pad(time, (0, pad_len), mode="edge")
        n = window_size

    flux_windows = []
    time_windows = []
    start = 0

    while start < n:
        end = start + window_size
        if end <= n:
            flux_windows.append(flux[start:end])
            time_windows.append(time[start:end])
        else:
            f_pad = np.pad(flux[start:], (0, end - n), mode="edge")
            t_pad = np.pad(time[start:], (0, end - n), mode="edge")
            flux_windows.append(f_pad)
            time_windows.append(t_pad)
        start += stride

        if start >= n:
            break

    flux_windows = np.array(flux_windows, dtype=np.float64)
    time_windows = np.array(time_windows, dtype=np.float64)

    logger.info(
        "Created %d windows (size=%d, stride=%d) from %d points.",
        len(flux_windows), window_size, stride, n,
    )
    return flux_windows, time_windows


def inject_transit(
    flux: np.ndarray,
    depth: float,
    center: int,
    duration: int,
) -> np.ndarray:
    flux = flux.copy()
    half = duration // 2
    start = max(center - half, 0)
    end = min(center + half, len(flux))

    n_pts = end - start
    if n_pts <= 0:
        return flux

    t = np.linspace(0.0, np.pi, n_pts)
    taper = 0.5 * (1.0 - np.cos(t))

    flux[start:end] -= depth * taper

    return flux


def generate_synthetic_dataset(
    n_samples: int = 5000,
    window_size: int = 2001,
    noise_sigma: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=42)

    depth_lo, depth_hi = config.TRANSIT_DEPTH_RANGE
    dur_frac_lo, dur_frac_hi = config.TRANSIT_DURATION_RANGE

    clean_curves = np.zeros((n_samples, window_size), dtype=np.float64)
    noisy_curves = np.zeros((n_samples, window_size), dtype=np.float64)
    labels = np.zeros(n_samples, dtype=np.int32)

    n_transit = n_samples // 2

    for i in range(n_samples):
        clean = np.zeros(window_size, dtype=np.float64)

        if i < n_transit:
            depth = rng.uniform(depth_lo, depth_hi)
            dur_frac = rng.uniform(dur_frac_lo, dur_frac_hi)
            duration = max(int(dur_frac * window_size), 5)
            center = rng.integers(duration, window_size - duration)
            clean = inject_transit(clean, depth=depth,
                                   center=center, duration=duration)
            labels[i] = 1

        clean_curves[i] = clean

        noise = rng.normal(0.0, noise_sigma, size=window_size)

        n_cycles = rng.uniform(1.0, 3.0)
        amplitude = rng.uniform(0.0002, 0.002)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        trend = amplitude * np.sin(
            2.0 * np.pi * n_cycles * np.arange(window_size) / window_size
            + phase
        )

        noisy_curves[i] = clean + noise + trend

    clean_curves = clean_curves[..., np.newaxis]
    noisy_curves = noisy_curves[..., np.newaxis]

    logger.info(
        "Generated %d synthetic samples (window=%d): %d transits, %d flat.",
        n_samples, window_size, n_transit, n_samples - n_transit,
    )
    return noisy_curves, clean_curves, labels


def prepare_classification_dataset(
    noisy: np.ndarray,
    clean: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        noisy,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )

    logger.info(
        "Split dataset → train=%d  test=%d  "
        "(transit ratio: train=%.2f, test=%.2f).",
        len(X_train), len(X_test),
        y_train.mean(), y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


def augment_with_noise(
    flux: np.ndarray,
    sigma_range: Tuple[float, float] = (0.0005, 0.003),
) -> np.ndarray:
    rng = np.random.default_rng()
    sigma = rng.uniform(*sigma_range)
    noise = rng.normal(0.0, sigma, size=flux.shape)
    augmented = flux + noise

    logger.debug("Augmented with Gaussian noise (σ=%.5f).", sigma)
    return augmented
