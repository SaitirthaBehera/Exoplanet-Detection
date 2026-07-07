"""
Optional trend removal (detrending) for light curves (Section 6).

Long-period stellar variability (spots, pulsations, instrumental drift)
can obscure a transit signal. This module provides three configurable
detrending methods that estimate and remove that slow-varying trend while
being deliberately tuned (via window/degree parameters) to leave short,
localized transit dips intact:

* **Polynomial fitting** — fit and subtract a low-degree polynomial trend
  across the whole light curve. Best for very smooth, slow drifts.
* **Savitzky–Golay filtering** — the method already used in
  ``src/data_pipeline.py``'s ``flatten_lightcurve``; a local polynomial
  smoother that tracks stellar variability while a sufficiently large
  window leaves short transit dips mostly untouched.
* **Running median** — a simple, very robust trend estimate; because a
  transit dip only affects a small fraction of points inside the window,
  the median is naturally resistant to being pulled down by the transit
  itself (as long as the window is wide enough).

Detrending is disabled by default (Section 6: "Allow detrending to be
enabled or disabled via configuration") because the two-stage AI models in
this project (denoising autoencoder + classifier) were trained on
un-detrended synthetic data; enabling this stage changes the flux
distribution the models see.

Example
-------
>>> from src.detrending import detrend
>>> flux_flat, trend = detrend(flux, method="savgol", savgol_window=401)
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

_VALID_METHODS = {"polynomial", "savgol", "running_median"}


def _detrend_polynomial(flux: np.ndarray, degree: int) -> np.ndarray:
    """Fit and evaluate a degree-N polynomial trend over sample index."""
    x = np.arange(len(flux), dtype=np.float64)
    finite_mask = np.isfinite(flux)
    if np.sum(finite_mask) <= degree:
        logger.warning(
            "Not enough finite points (%d) to fit a degree-%d polynomial; "
            "skipping detrend.", int(np.sum(finite_mask)), degree,
        )
        return np.full_like(flux, np.nanmedian(flux) if np.any(finite_mask) else 1.0)
    coeffs = np.polyfit(x[finite_mask], flux[finite_mask], deg=degree)
    trend = np.polyval(coeffs, x)
    return trend


def _detrend_savgol(flux: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """Estimate trend via Savitzky-Golay smoothing (matches data_pipeline.py's convention)."""
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(flux):
        window_length = len(flux) if len(flux) % 2 == 1 else len(flux) - 1
    if window_length <= polyorder:
        window_length = polyorder + 1 + (polyorder % 2 == 0)
    if window_length > len(flux) or window_length < 1:
        logger.warning(
            "Light curve too short (%d points) for Savitzky-Golay "
            "detrending; skipping.", len(flux),
        )
        return np.full_like(flux, np.nanmedian(flux))

    # savgol_filter cannot handle NaNs; fill temporarily with the median
    # for the purpose of trend estimation only (does not affect output flux).
    flux_filled = flux.copy()
    nan_mask = ~np.isfinite(flux_filled)
    if np.any(nan_mask):
        flux_filled[nan_mask] = np.nanmedian(flux_filled)

    trend = savgol_filter(flux_filled, window_length=window_length, polyorder=polyorder)
    return trend


def _detrend_running_median(flux: np.ndarray, window: int) -> np.ndarray:
    """Estimate trend via a running (rolling) median filter."""
    window = max(1, min(window, len(flux)))
    if window % 2 == 0:
        window += 1

    flux_filled = flux.copy()
    nan_mask = ~np.isfinite(flux_filled)
    if np.any(nan_mask):
        flux_filled[nan_mask] = np.nanmedian(flux_filled)

    trend = median_filter(flux_filled, size=window, mode="nearest")
    return trend


def detrend(
    flux: np.ndarray,
    method: str = "savgol",
    polynomial_degree: int = 3,
    savgol_window: int = 401,
    savgol_polyorder: int = 3,
    running_median_window: int = 101,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove long-term trends from a flux array while preserving transit dips.

    Parameters
    ----------
    flux:
        1-D flux array (typically already normalized so the baseline is
        near 1.0 or 0.0, though this function works on either convention).
    method:
        One of ``"polynomial"``, ``"savgol"`` (default), or
        ``"running_median"``.
    polynomial_degree:
        Degree used when ``method="polynomial"``. Keep this low
        (2-4) — high-degree polynomials can fit through and distort short
        transit dips.
    savgol_window, savgol_polyorder:
        Used when ``method="savgol"``. The window should span
        significantly more time than the longest expected transit duration
        so the dip is smoothed *around*, not *into*, the trend estimate.
    running_median_window:
        Used when ``method="running_median"``. Should likewise be wide
        relative to transit duration.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(flux_detrended, trend)`` — the flux with the trend divided out
        (``flux / trend``, matching the convention already used in
        ``src.data_pipeline.flatten_lightcurve``) and the estimated trend
        itself (useful for diagnostic plotting).

    Raises
    ------
    ValueError
        If ``method`` is not recognized.

    Notes
    -----
    Because a transit dip is short relative to stellar-variability
    timescales, all three methods are safe *as long as their
    window/degree parameters are chosen to be wide/low relative to the
    transit duration*. Very narrow windows or high polynomial degrees can
    fit through and partially cancel out real transit signals — callers
    should keep this in mind when tuning ``config/preprocessing.yaml``.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown detrending method '{method}'. "
            f"Expected one of {sorted(_VALID_METHODS)}."
        )

    flux = np.asarray(flux, dtype=np.float64)
    if flux.size == 0:
        logger.warning("Empty flux array passed to detrend(); returning as-is.")
        return flux.copy(), flux.copy()

    if method == "polynomial":
        trend = _detrend_polynomial(flux, degree=polynomial_degree)
    elif method == "savgol":
        trend = _detrend_savgol(flux, window_length=savgol_window, polyorder=savgol_polyorder)
    else:  # running_median
        trend = _detrend_running_median(flux, window=running_median_window)

    trend_safe = np.where(np.abs(trend) < 1e-10, 1.0, trend)
    flux_detrended = flux / trend_safe

    logger.info(
        "Detrended flux (method='%s'): trend range [%.6f, %.6f].",
        method, float(np.nanmin(trend)), float(np.nanmax(trend)),
    )

    return flux_detrended, trend
