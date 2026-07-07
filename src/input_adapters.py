"""
Input adapters with automatic format detection (Section 1).

Normalizes any of the following input types into a plain
``(time: np.ndarray, flux: np.ndarray, metadata: dict)`` tuple, which is
what every other preprocessing module in this package consumes:

1. ``lightkurve.LightCurve`` objects (Kepler / TESS, or any mission
   lightkurve supports)
2. FITS files (path to a light-curve FITS file)
3. CSV files (path to a CSV with time/flux columns)
4. Existing NumPy arrays (``(time, flux)`` tuple or a single 2-column array)
5. Existing Pandas DataFrames (with time/flux-like columns)

Lightkurve is treated as an *optional* dependency: it is only imported
inside the branch that needs it, so the rest of this module (and the rest
of the preprocessing pipeline) works fine in environments where
``lightkurve`` is not installed but FITS/CSV/array/DataFrame input is
still needed (e.g. this sandbox, CI, or a lightweight deployment).

Example
-------
>>> from src.input_adapters import load_input
>>> time, flux, meta = load_input("data/raw/some_star.csv")
>>> time, flux, meta = load_input((time_array, flux_array))
>>> time, flux, meta = load_input(some_dataframe, time_col="time", flux_col="flux")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Column name candidates searched (case-insensitively) when reading
# CSV/DataFrame input without explicit column names.
_TIME_COL_CANDIDATES = ["time", "bjd", "bkjd", "btjd", "t"]
_FLUX_COL_CANDIDATES = ["flux", "pdcsap_flux", "sap_flux", "f"]


def _find_column(columns, candidates) -> Optional[str]:
    """Case-insensitively find the first candidate name present in columns."""
    lower_map = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _from_arrays(
    time: np.ndarray, flux: np.ndarray, extra_metadata: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    time = np.asarray(time, dtype=np.float64).ravel()
    flux = np.asarray(flux, dtype=np.float64).ravel()
    if len(time) != len(flux):
        raise ValueError(
            f"time and flux must have equal length, got {len(time)} vs {len(flux)}."
        )
    metadata = {"source_type": "ndarray", **(extra_metadata or {})}
    return time, flux, metadata


def _load_from_lightkurve_object(lc) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract (time, flux, metadata) from a lightkurve.LightCurve-like object."""
    try:
        time = np.asarray(lc.time.value, dtype=np.float64)
    except AttributeError:
        # Some lightkurve versions / mocks expose time as a plain array.
        time = np.asarray(lc.time, dtype=np.float64)
    try:
        flux = np.asarray(lc.flux.value, dtype=np.float64)
    except AttributeError:
        flux = np.asarray(lc.flux, dtype=np.float64)

    metadata = {
        "source_type": "lightkurve",
        "target_id": getattr(lc, "targetid", None) or getattr(lc, "label", None),
        "mission": getattr(lc, "mission", None),
    }
    return time, flux, metadata


def _load_from_fits(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load ``(time, flux)`` from a Kepler/TESS-style light-curve FITS file.

    Requires ``astropy`` (already a project dependency for
    ``src/feature_extraction.py``). Looks for a ``TIME`` and
    ``PDCSAP_FLUX`` (falling back to ``SAP_FLUX``) column in the first
    binary table HDU, matching the standard Kepler/TESS light-curve FITS
    layout.
    """
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError(
            "Reading FITS files requires astropy. Install it via "
            "`pip install astropy` (see requirements.txt)."
        ) from exc

    with fits.open(path) as hdul:
        table_hdu = None
        for hdu in hdul:
            if hasattr(hdu, "columns") and hdu.columns is not None:
                names_upper = [n.upper() for n in hdu.columns.names]
                if "TIME" in names_upper:
                    table_hdu = hdu
                    break
        if table_hdu is None:
            raise ValueError(f"No light-curve table with a TIME column found in '{path}'.")

        data = table_hdu.data
        names_upper = {n.upper(): n for n in table_hdu.columns.names}
        time = np.asarray(data[names_upper["TIME"]], dtype=np.float64)

        flux_col = None
        for candidate in ["PDCSAP_FLUX", "SAP_FLUX", "FLUX"]:
            if candidate in names_upper:
                flux_col = names_upper[candidate]
                break
        if flux_col is None:
            raise ValueError(
                f"No recognized flux column (PDCSAP_FLUX/SAP_FLUX/FLUX) found in '{path}'."
            )
        flux = np.asarray(data[flux_col], dtype=np.float64)

        header = table_hdu.header
        metadata = {
            "source_type": "fits",
            "source_path": path,
            "flux_column": flux_col,
            "target_id": header.get("OBJECT"),
            "mission": header.get("TELESCOP"),
        }

    return time, flux, metadata


def _load_from_csv(
    path: str, time_col: Optional[str] = None, flux_col: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load ``(time, flux)`` from a CSV file, auto-detecting column names if not given."""
    import pandas as pd

    df = pd.read_csv(path)
    return _load_from_dataframe(df, time_col=time_col, flux_col=flux_col, source_path=path)


def _load_from_dataframe(
    df, time_col: Optional[str] = None, flux_col: Optional[str] = None,
    source_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load ``(time, flux)`` from a pandas DataFrame, auto-detecting columns if not given."""
    resolved_time_col = time_col or _find_column(df.columns, _TIME_COL_CANDIDATES)
    resolved_flux_col = flux_col or _find_column(df.columns, _FLUX_COL_CANDIDATES)

    if resolved_time_col is None or resolved_flux_col is None:
        raise ValueError(
            "Could not auto-detect time/flux columns in DataFrame with "
            f"columns {list(df.columns)}. Pass time_col= and flux_col= explicitly."
        )

    time = np.asarray(df[resolved_time_col].values, dtype=np.float64)
    flux = np.asarray(df[resolved_flux_col].values, dtype=np.float64)

    metadata = {
        "source_type": "dataframe",
        "time_column": resolved_time_col,
        "flux_column": resolved_flux_col,
    }
    if source_path:
        metadata["source_type"] = "csv"
        metadata["source_path"] = source_path

    return time, flux, metadata


def detect_input_type(data: Any) -> str:
    """
    Detect the type of a preprocessing input without fully parsing it.

    Parameters
    ----------
    data:
        Any of: a file path string (``.fits``/``.fits.gz`` or ``.csv``), a
        ``(time, flux)`` tuple/list of arrays, a 2-D NumPy array, a pandas
        DataFrame, or a lightkurve ``LightCurve``-like object (duck-typed
        via the presence of ``.time`` and ``.flux`` attributes).

    Returns
    -------
    str
        One of ``"fits"``, ``"csv"``, ``"ndarray_pair"``, ``"ndarray_2d"``,
        ``"dataframe"``, ``"lightkurve"``.

    Raises
    ------
    ValueError
        If the input type cannot be determined.
    """
    if isinstance(data, str):
        lower = data.lower()
        if lower.endswith((".fits", ".fits.gz", ".fit")):
            return "fits"
        if lower.endswith(".csv"):
            return "csv"
        raise ValueError(
            f"Cannot infer input type from file extension: '{data}'. "
            "Expected .fits or .csv."
        )

    # Duck-type lightkurve LightCurve objects: they expose .time and .flux
    # attributes but are not plain arrays/DataFrames/tuples.
    if hasattr(data, "time") and hasattr(data, "flux") and not isinstance(
        data, (tuple, list, np.ndarray)
    ):
        type_name = type(data).__module__ + "." + type(data).__name__
        if "pandas" not in type_name:
            return "lightkurve"

    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return "dataframe"
    except ImportError:
        pass

    if isinstance(data, (tuple, list)) and len(data) == 2:
        return "ndarray_pair"

    if isinstance(data, np.ndarray):
        if data.ndim == 2 and data.shape[1] == 2:
            return "ndarray_2d"
        raise ValueError(
            f"Cannot infer input type from a {data.ndim}-D array of shape "
            f"{data.shape}. Expected a 2-column (N, 2) array or a "
            "(time, flux) tuple of 1-D arrays."
        )

    raise ValueError(f"Cannot infer input type for object of type {type(data)}.")


def load_input(
    data: Any,
    time_col: Optional[str] = None,
    flux_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load and normalize any supported input type into ``(time, flux, metadata)``.

    Parameters
    ----------
    data:
        One of: a lightkurve ``LightCurve`` object, a path to a ``.fits``
        or ``.csv`` file, a ``(time, flux)`` tuple/list of 1-D arrays, a
        single ``(N, 2)`` NumPy array, or a pandas DataFrame.
    time_col, flux_col:
        Explicit column names to use for CSV/DataFrame input. If omitted,
        common names are auto-detected (see
        ``_TIME_COL_CANDIDATES``/``_FLUX_COL_CANDIDATES``).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
        ``(time, flux, metadata)`` — ``metadata`` always includes at least
        a ``"source_type"`` key describing which adapter handled the
        input, plus whatever provenance info that adapter could extract
        (e.g. FITS header target/mission, DataFrame column names used).

    Raises
    ------
    ValueError
        If the input type cannot be determined, required columns cannot
        be found, or ``time``/``flux`` lengths mismatch.
    ImportError
        If FITS input is requested but astropy is not installed.
    """
    input_type = detect_input_type(data)
    logger.info("Detected input type: '%s'.", input_type)

    if input_type == "fits":
        time, flux, metadata = _load_from_fits(data)
    elif input_type == "csv":
        time, flux, metadata = _load_from_csv(data, time_col=time_col, flux_col=flux_col)
    elif input_type == "dataframe":
        time, flux, metadata = _load_from_dataframe(data, time_col=time_col, flux_col=flux_col)
    elif input_type == "ndarray_pair":
        time, flux, metadata = _from_arrays(data[0], data[1])
    elif input_type == "ndarray_2d":
        time, flux, metadata = _from_arrays(data[:, 0], data[:, 1])
    elif input_type == "lightkurve":
        time, flux, metadata = _load_from_lightkurve_object(data)
    else:  # pragma: no cover — detect_input_type() guarantees one of the above
        raise ValueError(f"Unhandled input type '{input_type}'.")

    logger.info(
        "Loaded %d data point(s) via '%s' adapter.", len(flux), metadata["source_type"]
    )

    return time, flux, metadata
