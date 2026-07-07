"""
Window segmentation for light curves (Section 7).

Splits a full light curve into fixed-size, optionally overlapping windows
suitable for feeding to the downstream denoising autoencoder / classifier,
which expect fixed-length inputs. Each window carries its own time slice,
flux slice, a unique window ID, and metadata (source indices, padding
info) so windows remain traceable back to the original light curve.

This module's sliding-window mechanics mirror
``src.preprocessing.create_windows`` (same start/stride/edge-pad
behavior), but add: configurable overlap (instead of a raw stride),
per-window metadata records, and window IDs, per Section 7.

Example
-------
>>> from src.windowing import generate_windows
>>> windows = generate_windows(time, flux, window_size=2001, overlap=0.75)
>>> windows[0].window_id, windows[0].flux.shape
('window_0000', (2001,))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Window:
    """
    A single windowed segment of a light curve.

    Attributes
    ----------
    window_id:
        Unique identifier, e.g. ``"window_0000"``.
    time:
        1-D array of timestamps for this window, length == window_size.
    flux:
        1-D array of flux values for this window, length == window_size.
    metadata:
        Dict with provenance info: ``start_index``, ``end_index``
        (exclusive, before padding), ``n_padded`` (points added by
        padding), and any extra metadata passed in by the caller (e.g.
        target_id, mission).
    """
    window_id: str
    time: np.ndarray
    flux: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


def generate_windows(
    time: np.ndarray,
    flux: np.ndarray,
    window_size: int = 2001,
    overlap: float = 0.75,
    padding_mode: str = "edge",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Window]:
    """
    Segment a light curve into fixed-size, overlapping windows.

    Parameters
    ----------
    time, flux:
        1-D arrays of equal length.
    window_size:
        Number of samples per window. Default 2001 (matches the existing
        model input size in ``src/config.py``).
    overlap:
        Fraction of ``window_size`` that consecutive windows overlap by,
        in ``[0.0, 1.0)``. E.g. ``overlap=0.75`` with ``window_size=2001``
        produces a stride of ~500 samples — identical to the legacy
        ``create_windows(..., stride=500)`` default.
    padding_mode:
        ``numpy.pad`` mode used both to pad a light curve shorter than
        ``window_size`` and to pad the final (short) window at the end of
        a longer curve. Default ``"edge"`` (repeats the boundary value),
        matching existing project convention.
    extra_metadata:
        Optional dict merged into every window's ``metadata`` (e.g.
        ``{"target_id": "Kepler-10b", "mission": "Kepler"}``).

    Returns
    -------
    List[Window]
        One :class:`Window` per segment, each with a unique
        ``window_id``, its time/flux slice, and metadata describing its
        provenance (source index range, padding applied).

    Raises
    ------
    ValueError
        If ``window_size <= 0``, ``overlap`` is outside ``[0, 1)``, or
        ``time``/``flux`` lengths mismatch.

    Notes
    -----
    The sliding-window mechanics (stride computation, edge case where the
    final window would run past the array end, padding-when-necessary)
    match ``src.preprocessing.create_windows`` exactly when
    ``overlap = 1 - stride / window_size``, so this function is a strict,
    metadata-rich superset of the legacy windowing logic rather than a
    divergent reimplementation.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}.")
    if len(time) != len(flux):
        raise ValueError(
            f"time and flux must have equal length, got {len(time)} vs {len(flux)}."
        )

    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    extra_metadata = extra_metadata or {}

    stride = max(int(round(window_size * (1.0 - overlap))), 1)

    n_original = len(flux)
    n = n_original

    # Pad the whole light curve up to at least one window's worth of data.
    leading_pad = 0
    if n < window_size:
        pad_len = window_size - n
        flux = np.pad(flux, (0, pad_len), mode=padding_mode)
        time = np.pad(time, (0, pad_len), mode=padding_mode)
        n = window_size
        leading_pad = pad_len

    windows: List[Window] = []
    start = 0
    idx = 0

    # Stride based on the *original* (pre-padding) data length: once a
    # window's start position is beyond the real data, any further
    # "windows" would be pure padding and add no information.
    stride_limit = max(n_original, 1)

    while start < stride_limit:
        end = start + window_size
        if end <= n:
            flux_win = flux[start:end]
            time_win = time[start:end]
            n_padded = leading_pad if (start == 0 and leading_pad > 0) else 0
        else:
            flux_win = np.pad(flux[start:n], (0, end - n), mode=padding_mode)
            time_win = np.pad(time[start:n], (0, end - n), mode=padding_mode)
            n_padded = (end - n) + (leading_pad if start == 0 else 0)

        metadata = {
            "start_index": int(start),
            "end_index": int(min(end, n_original)),
            "n_padded": int(n_padded),
            **extra_metadata,
        }

        windows.append(
            Window(
                window_id=f"window_{idx:04d}",
                time=time_win,
                flux=flux_win,
                metadata=metadata,
            )
        )

        idx += 1
        start += stride
        if start >= stride_limit:
            break

    logger.info(
        "Generated %d window(s) (size=%d, overlap=%.2f, stride=%d) from %d points.",
        len(windows), window_size, overlap, stride, n_original,
    )

    return windows


def windows_to_arrays(windows: List[Window]) -> Dict[str, np.ndarray]:
    """
    Stack a list of :class:`Window` objects into flat arrays.

    Parameters
    ----------
    windows:
        List of windows as returned by :func:`generate_windows`.

    Returns
    -------
    Dict[str, np.ndarray]
        ``{"flux": (n_windows, window_size), "time": (n_windows, window_size),
        "window_ids": (n_windows,)}`` — convenient for feeding directly into
        model input pipelines that expect a stacked array (matching the
        return shape of the legacy ``create_windows``).
    """
    if not windows:
        return {
            "flux": np.empty((0, 0), dtype=np.float64),
            "time": np.empty((0, 0), dtype=np.float64),
            "window_ids": np.empty((0,), dtype=object),
        }

    flux_stack = np.stack([w.flux for w in windows]).astype(np.float64)
    time_stack = np.stack([w.time for w in windows]).astype(np.float64)
    ids = np.array([w.window_id for w in windows], dtype=object)

    return {"flux": flux_stack, "time": time_stack, "window_ids": ids}
