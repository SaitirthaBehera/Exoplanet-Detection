"""
Data validation for raw light-curve inputs (Section 2).

This module inspects a ``(time, flux)`` light curve *before* any
preprocessing is applied and produces a structured
:class:`ValidationReport` describing its quality: coverage, cadence,
missing values, duplicate timestamps, non-finite values, negative flux,
basic flux statistics, and an estimated noise level.

Design notes
------------
* This module is **read-only** — it never mutates the input arrays.
* Problems are reported as *warnings*, not exceptions, per Section 2
  ("If serious issues are found, generate warnings instead of immediately
  failing."). Callers can inspect ``report.warnings`` and ``report.is_valid``
  to decide whether to proceed, retry, or abort.
* This is deliberately independent from ``src/validation.py``, which
  validates *model detections* against NASA Exoplanet Archive parameters —
  a different concern (scientific result validation vs. input data QA).

Example
-------
>>> from src.data_validation import validate_lightcurve
>>> report = validate_lightcurve(time, flux, target_id="Kepler-10b", mission="Kepler")
>>> report.is_valid
True
>>> report.to_dict()["missing_value_pct"]
0.12
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """
    Structured report describing the quality of a raw light curve.

    Attributes
    ----------
    target_id:
        Identifier of the target star/planet (e.g. "Kepler-10b").
    mission:
        Originating mission, e.g. "Kepler" or "TESS".
    n_observations:
        Number of data points in the light curve.
    duration_days:
        Total observation span (max(time) - min(time)), in the same units
        as the input ``time`` array (days, for Kepler/TESS BTJD/BKJD).
    time_range:
        ``(min_time, max_time)`` tuple.
    median_cadence:
        Median spacing between consecutive (sorted) timestamps — a proxy
        for the instrument's sampling cadence.
    missing_value_pct:
        Percentage of flux samples that are NaN.
    n_duplicate_timestamps:
        Count of timestamps that occur more than once.
    n_infinite_values:
        Count of non-finite (±inf) flux samples.
    n_negative_flux:
        Count of negative flux samples (physically implausible for raw
        flux, though common after normalization — interpreted contextually
        by the caller).
    flux_stats:
        Dict of basic descriptive statistics: mean, median, std, min, max.
    estimated_noise:
        Robust noise estimate (median absolute deviation of the
        point-to-point flux differences, scaled to be a consistent
        estimator of the standard deviation under Gaussian noise).
    warnings:
        List of human-readable warning strings for any quality issues
        found. An empty list means no issues were detected.
    is_valid:
        ``False`` only when the light curve is unusable outright (e.g.
        empty, or all-NaN flux). Mere quality issues (gaps, duplicates,
        outliers) set warnings but leave this ``True`` — see module
        docstring.
    """
    target_id: str
    mission: str
    n_observations: int
    duration_days: float
    time_range: tuple
    median_cadence: float
    missing_value_pct: float
    n_duplicate_timestamps: int
    n_infinite_values: int
    n_negative_flux: int
    flux_stats: Dict[str, float]
    estimated_noise: float
    warnings: List[str] = field(default_factory=list)
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of this report."""
        d = asdict(self)
        d["time_range"] = list(d["time_range"])
        return d

    def summary(self) -> str:
        """Return a short, human-readable summary line."""
        status = "OK" if self.is_valid else "INVALID"
        return (
            f"[{status}] {self.target_id} ({self.mission}): "
            f"{self.n_observations} pts, {self.duration_days:.2f} d span, "
            f"{self.missing_value_pct:.2f}% missing, "
            f"{len(self.warnings)} warning(s)"
        )


def _estimate_noise(flux: np.ndarray) -> float:
    """
    Estimate the point-to-point noise level of a flux array.

    Uses the median absolute deviation (MAD) of first differences, scaled
    by 1.4826 / sqrt(2) so that it is a consistent estimator of the flux
    standard deviation under i.i.d. Gaussian noise, while being robust to
    outliers and slow trends (unlike a plain ``np.std``).

    Parameters
    ----------
    flux:
        Flux array. May contain NaNs; they are ignored.

    Returns
    -------
    float
        Estimated noise sigma. Returns 0.0 if fewer than 2 finite,
        consecutive-difference samples are available.
    """
    finite = flux[np.isfinite(flux)]
    if len(finite) < 2:
        return 0.0
    diffs = np.diff(finite)
    mad = np.median(np.abs(diffs - np.median(diffs)))
    # 1.4826 converts MAD -> sigma for a normal distribution;
    # /sqrt(2) accounts for differencing two independent samples.
    noise = float(mad * 1.4826 / np.sqrt(2.0))
    return noise


def validate_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    target_id: str = "unknown",
    mission: str = "unknown",
) -> ValidationReport:
    """
    Validate a raw ``(time, flux)`` light curve and build a quality report.

    Parameters
    ----------
    time:
        1-D array of timestamps (e.g. BKJD/BTJD days).
    flux:
        1-D array of flux values, same length as ``time``.
    target_id:
        Identifier for the target, used only for reporting/logging.
    mission:
        Mission name ("Kepler", "TESS", ...), used only for
        reporting/logging.

    Returns
    -------
    ValidationReport
        Structured report. See :class:`ValidationReport` for fields.

    Notes
    -----
    This function does not raise on data-quality problems — it records
    them as warnings. It only sets ``is_valid=False`` when the data is
    unusable outright (empty arrays, mismatched lengths, or all-NaN flux).
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    warnings_list: List[str] = []
    is_valid = True

    if time.shape != flux.shape:
        msg = (
            f"time and flux length mismatch: {time.shape} vs {flux.shape}."
        )
        logger.error(msg)
        warnings_list.append(msg)
        is_valid = False

    n_observations = int(len(flux))
    if n_observations == 0:
        msg = "Light curve is empty (0 observations)."
        logger.error(msg)
        warnings_list.append(msg)
        return ValidationReport(
            target_id=target_id,
            mission=mission,
            n_observations=0,
            duration_days=0.0,
            time_range=(float("nan"), float("nan")),
            median_cadence=float("nan"),
            missing_value_pct=100.0,
            n_duplicate_timestamps=0,
            n_infinite_values=0,
            n_negative_flux=0,
            flux_stats={},
            estimated_noise=0.0,
            warnings=warnings_list,
            is_valid=False,
        )

    # --- Missing values --------------------------------------------------
    n_missing = int(np.sum(np.isnan(flux)))
    missing_value_pct = 100.0 * n_missing / n_observations
    if missing_value_pct > 0:
        warnings_list.append(
            f"{missing_value_pct:.2f}% of flux values are missing (NaN)."
        )
    if missing_value_pct == 100.0:
        warnings_list.append("All flux values are missing (NaN).")
        is_valid = False

    # --- Infinite values ---------------------------------------------------
    n_infinite = int(np.sum(np.isinf(flux)))
    if n_infinite > 0:
        warnings_list.append(f"{n_infinite} infinite flux value(s) found.")

    # --- Negative flux -------------------------------------------------------
    finite_flux = flux[np.isfinite(flux)]
    n_negative = int(np.sum(finite_flux < 0)) if len(finite_flux) else 0
    if n_negative > 0:
        warnings_list.append(
            f"{n_negative} negative flux value(s) found "
            "(unexpected for raw, un-normalized flux)."
        )

    # --- Duplicate timestamps -------------------------------------------------
    finite_time = time[np.isfinite(time)]
    n_time_total = len(finite_time)
    n_unique_time = len(np.unique(finite_time)) if n_time_total else 0
    n_duplicate_timestamps = int(n_time_total - n_unique_time)
    if n_duplicate_timestamps > 0:
        warnings_list.append(
            f"{n_duplicate_timestamps} duplicate timestamp(s) found."
        )

    # --- Time range / duration / cadence ---------------------------------------
    if n_time_total >= 1:
        time_min = float(np.min(finite_time))
        time_max = float(np.max(finite_time))
        duration_days = float(time_max - time_min)
    else:
        time_min, time_max, duration_days = float("nan"), float("nan"), 0.0
        warnings_list.append("No finite timestamps available.")
        is_valid = False

    if n_time_total >= 2:
        sorted_time = np.sort(finite_time)
        cadence_diffs = np.diff(sorted_time)
        cadence_diffs = cadence_diffs[cadence_diffs > 0]  # exclude duplicates
        median_cadence = float(np.median(cadence_diffs)) if len(cadence_diffs) else 0.0
    else:
        median_cadence = 0.0
        warnings_list.append("Insufficient timestamps to estimate cadence.")

    # Detect large gaps (>10x median cadence) as an informational warning.
    if median_cadence > 0 and n_time_total >= 2:
        sorted_time = np.sort(finite_time)
        gaps = np.diff(sorted_time)
        n_large_gaps = int(np.sum(gaps > 10 * median_cadence))
        if n_large_gaps > 0:
            warnings_list.append(
                f"{n_large_gaps} large observation gap(s) "
                f"(>10x median cadence) detected."
            )

    # --- Flux statistics ---------------------------------------------------
    if len(finite_flux) > 0:
        flux_stats = {
            "mean": float(np.mean(finite_flux)),
            "median": float(np.median(finite_flux)),
            "std": float(np.std(finite_flux)),
            "min": float(np.min(finite_flux)),
            "max": float(np.max(finite_flux)),
        }
    else:
        flux_stats = {}
        warnings_list.append("No finite flux values available for statistics.")
        is_valid = False

    # --- Noise estimate ------------------------------------------------------
    estimated_noise = _estimate_noise(flux)

    report = ValidationReport(
        target_id=target_id,
        mission=mission,
        n_observations=n_observations,
        duration_days=duration_days,
        time_range=(time_min, time_max),
        median_cadence=median_cadence,
        missing_value_pct=missing_value_pct,
        n_duplicate_timestamps=n_duplicate_timestamps,
        n_infinite_values=n_infinite,
        n_negative_flux=n_negative,
        flux_stats=flux_stats,
        estimated_noise=estimated_noise,
        warnings=warnings_list,
        is_valid=is_valid,
    )

    log_level = logging.INFO if is_valid else logging.ERROR
    logger.log(log_level, report.summary())
    for w in warnings_list:
        logger.warning("  - %s", w)

    return report
