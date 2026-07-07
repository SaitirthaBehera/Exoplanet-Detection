"""
Per-light-curve feature statistics (Section 9).

Computes a fixed set of descriptive statistics for a processed light
curve, intended to be stored alongside the processed data
(``statistics.json``) for later visualization/dashboarding — distinct
from the transit-specific features computed by
``src.feature_extraction`` (period, depth, duration, SNR from BLS), which
are scientific detection outputs rather than general data-quality
statistics.

Example
-------
>>> from src.feature_statistics import compute_statistics
>>> stats = compute_statistics(time, flux)
>>> stats.to_dict()["rms"]
0.0234
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LightCurveStatistics:
    """
    Descriptive statistics for a single processed light curve.

    Attributes
    ----------
    mean_flux, median_flux, std_flux, variance_flux, min_flux, max_flux:
        Standard descriptive statistics of the flux array (NaNs excluded).
    rms:
        Root-mean-square of the flux array — sensitive to both the signal
        level and its variability.
    estimated_noise:
        Robust point-to-point noise estimate (see
        ``src.data_validation._estimate_noise`` for the same method,
        reused here for consistency).
    n_samples:
        Total number of samples in the light curve (including any NaNs).
    observation_duration:
        ``max(time) - min(time)`` over finite timestamps.
    missing_value_count:
        Count of NaN flux samples.
    gap_count:
        Count of "gaps" — consecutive-timestamp spacings greater than 10x
        the median cadence (same heuristic used in
        ``src.data_validation.validate_lightcurve``, kept consistent so
        the two reports agree).
    """
    mean_flux: float
    median_flux: float
    std_flux: float
    variance_flux: float
    min_flux: float
    max_flux: float
    rms: float
    estimated_noise: float
    n_samples: int
    observation_duration: float
    missing_value_count: int
    gap_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return asdict(self)


def _estimate_noise(flux: np.ndarray) -> float:
    """Robust point-to-point noise estimate (MAD of first differences)."""
    finite = flux[np.isfinite(flux)]
    if len(finite) < 2:
        return 0.0
    diffs = np.diff(finite)
    mad = np.median(np.abs(diffs - np.median(diffs)))
    return float(mad * 1.4826 / np.sqrt(2.0))


def _count_gaps(time: np.ndarray) -> int:
    """Count gaps larger than 10x the median cadence."""
    finite_time = time[np.isfinite(time)]
    if len(finite_time) < 2:
        return 0
    sorted_time = np.sort(finite_time)
    diffs = np.diff(sorted_time)
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        return 0
    median_cadence = float(np.median(positive_diffs))
    if median_cadence <= 0:
        return 0
    return int(np.sum(diffs > 10 * median_cadence))


def compute_statistics(
    time: np.ndarray,
    flux: np.ndarray,
) -> LightCurveStatistics:
    """
    Compute the full set of Section-9 descriptive statistics for a light curve.

    Parameters
    ----------
    time, flux:
        1-D arrays of equal length. May contain NaNs — they are excluded
        from flux statistics but counted via ``missing_value_count``.

    Returns
    -------
    LightCurveStatistics
        Populated statistics object. All-NaN or empty input yields zeros
        for numeric fields (with a logged warning) rather than raising.

    Raises
    ------
    ValueError
        If ``time`` and ``flux`` have mismatched lengths.
    """
    if len(time) != len(flux):
        raise ValueError(
            f"time and flux must have equal length, got {len(time)} vs {len(flux)}."
        )

    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    n_samples = int(len(flux))

    finite_flux_mask = np.isfinite(flux)
    finite_flux = flux[finite_flux_mask]
    missing_value_count = int(np.sum(~finite_flux_mask))

    if len(finite_flux) == 0:
        logger.warning(
            "No finite flux values available; returning zeroed statistics."
        )
        stats = LightCurveStatistics(
            mean_flux=0.0, median_flux=0.0, std_flux=0.0, variance_flux=0.0,
            min_flux=0.0, max_flux=0.0, rms=0.0, estimated_noise=0.0,
            n_samples=n_samples, observation_duration=0.0,
            missing_value_count=missing_value_count, gap_count=0,
        )
        return stats

    finite_time = time[np.isfinite(time)]
    observation_duration = (
        float(np.max(finite_time) - np.min(finite_time)) if len(finite_time) else 0.0
    )

    stats = LightCurveStatistics(
        mean_flux=float(np.mean(finite_flux)),
        median_flux=float(np.median(finite_flux)),
        std_flux=float(np.std(finite_flux)),
        variance_flux=float(np.var(finite_flux)),
        min_flux=float(np.min(finite_flux)),
        max_flux=float(np.max(finite_flux)),
        rms=float(np.sqrt(np.mean(np.square(finite_flux)))),
        estimated_noise=_estimate_noise(flux),
        n_samples=n_samples,
        observation_duration=observation_duration,
        missing_value_count=missing_value_count,
        gap_count=_count_gaps(time),
    )

    logger.info(
        "Computed statistics: n=%d, mean=%.6f, std=%.6f, rms=%.6f, "
        "noise=%.6f, missing=%d, gaps=%d.",
        stats.n_samples, stats.mean_flux, stats.std_flux, stats.rms,
        stats.estimated_noise, stats.missing_value_count, stats.gap_count,
    )

    return stats
