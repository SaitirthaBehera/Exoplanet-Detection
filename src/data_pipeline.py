import logging
import os
from typing import Dict, Optional, Tuple

import lightkurve as lk
import numpy as np
from scipy.signal import savgol_filter

from src import config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


def download_lightcurve(
    target_name: str,
    mission: str = "Kepler",
    author: str = "Kepler",
) -> Tuple[np.ndarray, np.ndarray]:
    logger.info("Searching MAST for '%s' (mission=%s, author=%s) …",
                target_name, mission, author)

    search_result = lk.search_lightcurve(
        target_name, mission=mission, author=author,
    )
    if len(search_result) == 0:
        raise RuntimeError(
            f"No light-curve products found for '{target_name}' "
            f"(mission={mission}, author={author})."
        )

    logger.info("Found %d quarter(s)/sector(s). Downloading …",
                len(search_result))

    # Robust download loop to auto-clean corrupt cached FITS files
    import re
    max_attempts = 15
    lc_collection = None
    for attempt in range(max_attempts):
        try:
            lc_collection = search_result.download_all(quality_bitmask="default")
            break
        except lk.utils.LightkurveError as e:
            msg = str(e)
            match = re.search(r"Error in reading Data product (.*?) of type", msg)
            if match:
                path = match.group(1).strip()
                if os.path.exists(path):
                    logger.warning("Removing corrupted file and retrying: %s", path)
                    try:
                        os.remove(path)
                    except Exception as rm_err:
                        logger.error("Failed to remove file %s: %s", path, rm_err)
                else:
                    logger.warning("Corrupted file path parsed but not found: %s", path)
            else:
                logger.warning("LightkurveError occurred but path not matched: %s. Retrying...", msg)
        except Exception as e:
            logger.warning("Error during download: %s. Retrying...", e)
    
    if lc_collection is None:
        lc_collection = search_result.download_all(quality_bitmask="default")

    stitched_lc = lc_collection.stitch(
        corrector_func=lambda x: x.remove_nans().normalize(),
    )

    time = np.asarray(stitched_lc.time.value, dtype=np.float64)
    flux = np.asarray(stitched_lc.flux.value, dtype=np.float64)

    logger.info("Downloaded %d data points for '%s'.", len(time), target_name)
    return time, flux


def clean_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(time) & np.isfinite(flux)
    time = time[finite_mask]
    flux = flux[finite_mask]
    n_removed_nan = int(np.sum(~finite_mask))

    median_flux = np.median(flux)
    std_flux = np.std(flux)
    clip_mask = np.abs(flux - median_flux) < sigma * std_flux
    time_clean = time[clip_mask]
    flux_clean = flux[clip_mask]
    n_removed_sigma = int(np.sum(~clip_mask))

    logger.info(
        "Cleaned light curve: removed %d NaN/Inf + %d sigma-clip outliers "
        "→ %d points remain.",
        n_removed_nan, n_removed_sigma, len(time_clean),
    )
    return time_clean, flux_clean


def flatten_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    window_length: int = 401,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(flux):
        window_length = len(flux) if len(flux) % 2 == 1 else len(flux) - 1

    trend = savgol_filter(flux, window_length=window_length, polyorder=3)

    trend_safe = np.where(np.abs(trend) < 1e-10, 1.0, trend)
    flattened_flux = flux / trend_safe

    logger.info(
        "Flattened light curve (window=%d): trend range [%.6f, %.6f].",
        window_length, trend.min(), trend.max(),
    )
    return time, flattened_flux, trend


def export_lightcurve(
    target_name: str,
    time: np.ndarray,
    flux: np.ndarray,
    output_dir: Optional[str] = None,
) -> str:
    if output_dir is None:
        output_dir = config.DATA_RAW_DIR

    os.makedirs(output_dir, exist_ok=True)

    data = np.column_stack([time, flux])
    filename = f"{target_name}.npy"
    filepath = os.path.join(output_dir, filename)
    np.save(filepath, data)

    logger.info("Exported '%s' → %s  (shape %s)", target_name, filepath,
                data.shape)
    return filepath


def download_all_targets() -> Dict[str, str]:
    results: Dict[str, str] = {}

    for target_name, meta in config.TARGETS.items():
        search_name = meta["search_name"]
        mission = meta.get("mission", "Kepler")
        logger.info("=" * 60)
        logger.info("Processing target: %s (%s)", target_name, search_name)
        logger.info("=" * 60)

        try:
            time, flux = download_lightcurve(
                target_name=search_name,
                mission=mission,
                author=mission,
            )

            time, flux = clean_lightcurve(
                time, flux, sigma=config.SIGMA_CLIP,
            )

            time, flux, _trend = flatten_lightcurve(
                time, flux, window_length=config.FLATTEN_WINDOW,
            )

            filepath = export_lightcurve(target_name, time, flux)
            results[target_name] = filepath

        except Exception:
            logger.exception(
                "Failed to process target '%s'. Skipping.", target_name,
            )

    logger.info(
        "Batch download complete: %d / %d targets succeeded.",
        len(results), len(config.TARGETS),
    )
    return results


def load_lightcurve(
    target_name: str,
    data_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if data_dir is None:
        data_dir = config.DATA_RAW_DIR

    filepath = os.path.join(data_dir, f"{target_name}.npy")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Light-curve file not found: {filepath}"
        )

    data = np.load(filepath)
    time = data[:, 0]
    flux = data[:, 1]

    logger.info("Loaded '%s' from %s  (shape %s)", target_name, filepath,
                data.shape)
    return time, flux
