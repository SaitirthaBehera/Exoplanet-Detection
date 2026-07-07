"""
Missing-value handling for light-curve flux arrays (Section 3).

Provides configurable strategies to fill NaN gaps in a flux array:
linear, cubic, and nearest-neighbor interpolation, plus forward-fill and
backward-fill. All strategies operate on ``(time, flux)`` pairs so that
gaps are filled with respect to the actual (possibly irregular) timestamps
rather than assuming uniform cadence.

Example
-------
>>> from src.missing_values import handle_missing_values
>>> filled_flux, n_filled = handle_missing_values(time, flux, strategy="linear")
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = {"linear", "cubic", "nearest", "ffill", "bfill"}


def handle_missing_values(
    time: np.ndarray,
    flux: np.ndarray,
    strategy: str = "linear",
) -> Tuple[np.ndarray, int]:
    """
    Fill missing (NaN) values in a flux array using a configurable strategy.

    Parameters
    ----------
    time:
        1-D array of timestamps, same length as ``flux``. Used as the
        interpolation x-axis so gaps are filled with respect to true time
        spacing rather than sample index.
    flux:
        1-D flux array, possibly containing NaNs.
    strategy:
        One of ``"linear"``, ``"cubic"``, ``"nearest"``, ``"ffill"``
        (forward fill), or ``"bfill"`` (backward fill).

    Returns
    -------
    Tuple[np.ndarray, int]
        ``(filled_flux, n_interpolated)`` — the flux array with NaNs
        filled, and a count of how many values were filled.

    Raises
    ------
    ValueError
        If ``strategy`` is not one of the supported options, or if
        ``time``/``flux`` have mismatched lengths.

    Notes
    -----
    * If there are no missing values, the input flux is returned unchanged
      (as a copy) and ``n_interpolated`` is 0.
    * If *all* values are missing, filling is impossible; the original
      (all-NaN) array is returned and a warning is logged — callers should
      check :func:`src.data_validation.validate_lightcurve` beforehand to
      catch this case.
    * ``cubic``/``nearest``/``linear`` use :class:`scipy.interpolate.interp1d`
      with the two boundary NaN regions filled by edge-value extrapolation
      (equivalent to ffill/bfill at the boundaries) so that interpolation
      never introduces new NaNs at the array edges.
    """
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unknown missing-value strategy '{strategy}'. "
            f"Expected one of {sorted(_VALID_STRATEGIES)}."
        )
    if len(time) != len(flux):
        raise ValueError(
            f"time and flux must have equal length, got {len(time)} vs {len(flux)}."
        )

    flux = np.asarray(flux, dtype=np.float64).copy()
    time = np.asarray(time, dtype=np.float64)

    nan_mask = np.isnan(flux)
    n_missing = int(np.sum(nan_mask))

    if n_missing == 0:
        logger.info("No missing values found; nothing to interpolate.")
        return flux, 0

    valid_mask = ~nan_mask
    n_valid = int(np.sum(valid_mask))

    if n_valid == 0:
        logger.warning(
            "Cannot fill missing values: all %d flux samples are NaN.",
            len(flux),
        )
        return flux, 0

    if n_valid < 2 and strategy in ("linear", "cubic", "nearest"):
        logger.warning(
            "Fewer than 2 valid samples available; falling back to "
            "forward/backward fill for strategy '%s'.",
            strategy,
        )
        strategy = "ffill"

    if strategy in ("linear", "cubic", "nearest"):
        kind = strategy
        if strategy == "cubic" and n_valid < 4:
            logger.warning(
                "Cubic interpolation requires >= 4 valid points "
                "(found %d); falling back to linear.", n_valid
            )
            kind = "linear"

        f = interp1d(
            time[valid_mask],
            flux[valid_mask],
            kind=kind,
            bounds_error=False,
            fill_value=(flux[valid_mask][0], flux[valid_mask][-1]),
        )
        flux[nan_mask] = f(time[nan_mask])

    elif strategy == "ffill":
        last_valid = None
        for i in range(len(flux)):
            if not nan_mask[i]:
                last_valid = flux[i]
            elif last_valid is not None:
                flux[i] = last_valid
        # Any leading NaNs (no prior valid value) get backward-filled.
        if np.isnan(flux[0]):
            first_valid_idx = np.argmax(valid_mask)
            flux[: first_valid_idx] = flux[first_valid_idx]

    elif strategy == "bfill":
        next_valid = None
        for i in range(len(flux) - 1, -1, -1):
            if not nan_mask[i]:
                next_valid = flux[i]
            elif next_valid is not None:
                flux[i] = next_valid
        # Any trailing NaNs (no later valid value) get forward-filled.
        if np.isnan(flux[-1]):
            last_valid_idx = len(flux) - 1 - np.argmax(valid_mask[::-1])
            flux[last_valid_idx:] = flux[last_valid_idx]

    n_remaining_nan = int(np.sum(np.isnan(flux)))
    n_interpolated = n_missing - n_remaining_nan

    logger.info(
        "Missing-value handling (strategy='%s'): filled %d/%d missing "
        "value(s); %d remain unfilled.",
        strategy, n_interpolated, n_missing, n_remaining_nan,
    )

    return flux, n_interpolated
