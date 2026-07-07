"""
Outlier detection and removal for light-curve flux arrays (Section 4).

Provides three configurable outlier-detection methods:

* **Sigma clipping** — flag points more than N standard deviations from the
  median (default, 5-sigma).
* **Median Absolute Deviation (MAD)** — a robust alternative to sigma
  clipping that is less sensitive to the outliers it's trying to detect.
* **Percentile clipping** — flag points outside a given percentile range.

Example
-------
>>> from src.outlier_detection import remove_outliers
>>> time_c, flux_c, stats = remove_outliers(time, flux, method="sigma_clip", sigma_threshold=5.0)
>>> stats["n_removed"], stats["pct_removed"]
(12, 0.24)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

_VALID_METHODS = {"sigma_clip", "mad", "percentile"}


@dataclass
class OutlierStats:
    """Summary statistics from an outlier-removal pass."""
    method: str
    n_input: int
    n_removed: int
    pct_removed: float

    def to_dict(self) -> dict:
        return asdict(self)


def _sigma_clip_mask(flux: np.ndarray, sigma_threshold: float) -> np.ndarray:
    """Return a boolean keep-mask using median +/- sigma_threshold * std."""
    median = np.median(flux)
    std = np.std(flux)
    if std == 0:
        logger.warning("Flux standard deviation is zero; sigma clipping skipped.")
        return np.ones_like(flux, dtype=bool)
    return np.abs(flux - median) <= sigma_threshold * std


def _mad_mask(flux: np.ndarray, mad_threshold: float) -> np.ndarray:
    """
    Return a boolean keep-mask using the modified z-score (MAD-based).

    modified_z = 0.6745 * (x - median) / MAD
    Points with |modified_z| > mad_threshold are flagged as outliers.
    0.6745 is the constant that makes MAD a consistent estimator of sigma
    for normally distributed data.
    """
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    if mad == 0:
        logger.warning("MAD is zero; MAD-based outlier detection skipped.")
        return np.ones_like(flux, dtype=bool)
    modified_z = 0.6745 * (flux - median) / mad
    return np.abs(modified_z) <= mad_threshold


def _percentile_mask(
    flux: np.ndarray, percentile_lower: float, percentile_upper: float
) -> np.ndarray:
    """Return a boolean keep-mask using a [lower, upper] percentile range."""
    lo = np.percentile(flux, percentile_lower)
    hi = np.percentile(flux, percentile_upper)
    return (flux >= lo) & (flux <= hi)


def remove_outliers(
    time: np.ndarray,
    flux: np.ndarray,
    method: str = "sigma_clip",
    sigma_threshold: float = 5.0,
    mad_threshold: float = 3.5,
    percentile_lower: float = 0.5,
    percentile_upper: float = 99.5,
) -> Tuple[np.ndarray, np.ndarray, OutlierStats]:
    """
    Detect and remove outliers from a light curve using a configurable method.

    Parameters
    ----------
    time, flux:
        1-D arrays of equal length. NaNs in ``flux`` are treated as
        already-invalid and are excluded from the kept output (they are
        not counted as "removed outliers", just passed through the
        finite-value filter).
    method:
        One of ``"sigma_clip"`` (default), ``"mad"``, or ``"percentile"``.
    sigma_threshold:
        Used when ``method="sigma_clip"``. Points beyond this many standard
        deviations from the median are removed. Default 5.0 (per spec).
    mad_threshold:
        Used when ``method="mad"``. Points with a modified z-score beyond
        this threshold are removed. Default 3.5 (a common convention).
    percentile_lower, percentile_upper:
        Used when ``method="percentile"``. Points outside
        ``[percentile_lower, percentile_upper]`` are removed.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, OutlierStats]
        ``(time_clean, flux_clean, stats)`` where ``stats`` reports the
        method used, input size, number removed, and percent removed.

    Raises
    ------
    ValueError
        If ``method`` is not recognized, or ``time``/``flux`` lengths
        mismatch.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown outlier method '{method}'. Expected one of {sorted(_VALID_METHODS)}."
        )
    if len(time) != len(flux):
        raise ValueError(
            f"time and flux must have equal length, got {len(time)} vs {len(flux)}."
        )

    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    n_input = len(flux)

    finite_mask = np.isfinite(time) & np.isfinite(flux)
    if not np.any(finite_mask):
        logger.warning("No finite flux/time values available for outlier detection.")
        stats = OutlierStats(method=method, n_input=n_input, n_removed=n_input, pct_removed=100.0)
        return time[finite_mask], flux[finite_mask], stats

    finite_flux = flux[finite_mask]

    if method == "sigma_clip":
        keep_within_finite = _sigma_clip_mask(finite_flux, sigma_threshold)
    elif method == "mad":
        keep_within_finite = _mad_mask(finite_flux, mad_threshold)
    else:  # percentile
        keep_within_finite = _percentile_mask(
            finite_flux, percentile_lower, percentile_upper
        )

    # Compose the finite-mask and the outlier keep-mask into one full-length mask.
    keep_mask = np.zeros(n_input, dtype=bool)
    keep_mask[finite_mask] = keep_within_finite

    time_clean = time[keep_mask]
    flux_clean = flux[keep_mask]

    n_removed = int(n_input - len(flux_clean))
    pct_removed = 100.0 * n_removed / n_input if n_input > 0 else 0.0

    stats = OutlierStats(
        method=method,
        n_input=n_input,
        n_removed=n_removed,
        pct_removed=pct_removed,
    )

    logger.info(
        "Outlier removal (method='%s'): removed %d/%d points (%.2f%%).",
        method, n_removed, n_input, pct_removed,
    )

    return time_clean, flux_clean, stats
