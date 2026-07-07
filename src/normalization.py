"""
Flux normalization for light curves (Section 5).

Provides four configurable normalization methods:

* **Min-Max scaling** — rescale flux to ``[0, 1]``.
* **Z-score normalization** — zero mean, unit variance.
* **Median normalization** — the convention used elsewhere in this project
  (``flux / median - 1``), centering the out-of-transit baseline at 0.
* **Robust scaling** — center on the median, scale by the interquartile
  range (IQR); robust to outliers that survived outlier removal.

Every method returns the fitted parameters alongside the normalized flux
so that the exact transform can be reproduced or inverted later (Section 5:
"Store normalization parameters for reproducibility").

Example
-------
>>> from src.normalization import normalize
>>> flux_norm, params = normalize(flux, method="median")
>>> params
{'method': 'median', 'median': 1.00021, ...}
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_VALID_METHODS = {"minmax", "zscore", "median", "robust"}


def _normalize_minmax(flux: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    f_min = float(np.min(flux))
    f_max = float(np.max(flux))
    span = f_max - f_min
    if span == 0:
        logger.warning("Flux range is zero; min-max normalization skipped.")
        return flux.copy(), {"method": "minmax", "min": f_min, "max": f_max}
    normalized = (flux - f_min) / span
    return normalized, {"method": "minmax", "min": f_min, "max": f_max}


def _normalize_zscore(flux: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    mean = float(np.mean(flux))
    std = float(np.std(flux))
    if std == 0:
        logger.warning("Flux standard deviation is zero; z-score normalization skipped.")
        return flux.copy() - mean, {"method": "zscore", "mean": mean, "std": std}
    normalized = (flux - mean) / std
    return normalized, {"method": "zscore", "mean": mean, "std": std}


def _normalize_median(flux: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    median = float(np.median(flux))
    if np.abs(median) < 1e-15:
        logger.warning("Median flux is near zero (%.3e); skipping division.", median)
        return flux.copy(), {"method": "median", "median": median}
    normalized = (flux / median) - 1.0
    return normalized, {"method": "median", "median": median}


def _normalize_robust(flux: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    median = float(np.median(flux))
    q1 = float(np.percentile(flux, 25))
    q3 = float(np.percentile(flux, 75))
    iqr = q3 - q1
    if iqr == 0:
        logger.warning("IQR is zero; robust scaling skipped (centering only).")
        return flux.copy() - median, {
            "method": "robust", "median": median, "q1": q1, "q3": q3, "iqr": iqr
        }
    normalized = (flux - median) / iqr
    return normalized, {"method": "robust", "median": median, "q1": q1, "q3": q3, "iqr": iqr}


_METHOD_DISPATCH = {
    "minmax": _normalize_minmax,
    "zscore": _normalize_zscore,
    "median": _normalize_median,
    "robust": _normalize_robust,
}


def normalize(
    flux: np.ndarray,
    method: str = "median",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize a flux array using a configurable method.

    Parameters
    ----------
    flux:
        1-D flux array. Should be free of NaN/Inf (run missing-value
        handling and outlier removal first); non-finite values will
        distort the fitted statistics if present.
    method:
        One of ``"minmax"``, ``"zscore"``, ``"median"`` (default), or
        ``"robust"``.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        ``(normalized_flux, params)`` — the normalized array and a dict of
        the fitted parameters (method name plus whatever statistics define
        the transform), suitable for JSON serialization and for later
        reproducing or inverting the normalization.

    Raises
    ------
    ValueError
        If ``method`` is not recognized.

    Notes
    -----
    The ``"median"`` method matches the convention already used by
    :func:`src.preprocessing.normalize_flux` (``flux / median - 1``), so
    swapping between the legacy function and this configurable one for the
    default method produces identical numerical output.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown normalization method '{method}'. "
            f"Expected one of {sorted(_VALID_METHODS)}."
        )

    flux = np.asarray(flux, dtype=np.float64)
    if flux.size == 0:
        logger.warning("Empty flux array passed to normalize(); returning as-is.")
        return flux.copy(), {"method": method}

    normalized, params = _METHOD_DISPATCH[method](flux)

    logger.info(
        "Normalized flux using method='%s' → output range [%.6f, %.6f].",
        method, float(np.min(normalized)), float(np.max(normalized)),
    )

    return normalized, params
