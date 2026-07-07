import logging
from typing import Dict, Optional, Tuple

import numpy as np
from astropy.timeseries import BoxLeastSquares
import astropy.units as u

from src.config import (
    BLS_PERIOD_MIN,
    BLS_PERIOD_MAX,
    BLS_PERIOD_NPOINTS,
    BLS_DURATIONS,
)

logger = logging.getLogger(__name__)


def run_bls_search(
    time: np.ndarray,
    flux: np.ndarray,
    period_min: float = BLS_PERIOD_MIN,
    period_max: float = BLS_PERIOD_MAX,
    n_periods: int = BLS_PERIOD_NPOINTS,
) -> Dict:
    if len(time) == 0 or len(flux) == 0:
        raise ValueError("time and flux arrays must be non-empty.")
    if len(time) != len(flux):
        raise ValueError(
            f"time and flux must have equal length, "
            f"got {len(time)} vs {len(flux)}."
        )

    logger.info(
        "Running BLS search — %d data points, period grid [%.2f, %.2f] d "
        "with %d trial periods",
        len(time), period_min, period_max, n_periods,
    )

    try:
        bls = BoxLeastSquares(time * u.day, flux)

        periods = np.linspace(period_min, period_max, n_periods)
        durations = np.array(BLS_DURATIONS)

        results = bls.power(periods * u.day, durations * u.day)

        power = np.asarray(results.power)
        best_idx = np.argmax(power)

        best_period = float(periods[best_idx])
        best_t0 = float(results.transit_time[best_idx].value)
        best_duration = float(results.duration[best_idx].value)
        best_depth = float(results.depth[best_idx])

        logger.info(
            "BLS result — best_period=%.5f d, depth=%.6f, "
            "duration=%.4f d, t0=%.4f",
            best_period, best_depth, best_duration, best_t0,
        )

        return {
            "best_period": best_period,
            "best_t0": best_t0,
            "best_duration": best_duration,
            "best_depth": best_depth,
            "bls_power": power,
            "periods": periods,
        }

    except Exception as exc:
        logger.error("BLS search failed: %s", exc)
        raise


def extract_transit_params(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
) -> Dict:
    try:
        phase, folded_flux = fold_lightcurve(time, flux, period, t0)

        oot_mask = np.abs(phase) > 0.15
        if np.sum(oot_mask) < 10:
            oot_mask = np.abs(phase) > 0.25

        baseline = np.median(folded_flux[oot_mask])

        transit_mask = np.abs(phase) < 0.05
        if np.sum(transit_mask) < 3:
            transit_mask = np.abs(phase) < 0.10

        transit_min = np.min(folded_flux[transit_mask]) if np.any(transit_mask) else baseline
        depth = float(baseline - transit_min)

        half_depth = baseline - 0.5 * depth
        in_transit = folded_flux < half_depth
        if np.any(in_transit):
            transit_phases = phase[in_transit]
            duration_phase = float(np.max(transit_phases) - np.min(transit_phases))
        else:
            duration_phase = 0.0

        duration_hours = duration_phase * period * 24.0

        transit_full_mask = np.abs(phase) < 0.05
        snr = compute_snr(folded_flux, transit_full_mask)

        time_span = float(np.max(time) - np.min(time))
        n_transits = max(1, int(np.floor(time_span / period)))

        logger.info(
            "Transit params — depth=%.6f, duration=%.2f h, "
            "snr=%.2f, n_transits=%d",
            depth, duration_hours, snr, n_transits,
        )

        return {
            "depth": depth,
            "duration_hours": duration_hours,
            "snr": snr,
            "n_transits": n_transits,
        }

    except Exception as exc:
        logger.error("Transit parameter extraction failed: %s", exc)
        raise


def fold_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    phase = ((time - t0) / period) % 1.0

    phase[phase > 0.5] -= 1.0

    sort_idx = np.argsort(phase)
    phase = phase[sort_idx]
    folded_flux = flux[sort_idx]

    return phase, folded_flux


def compute_snr(flux: np.ndarray, transit_mask: np.ndarray) -> float:
    oot_mask = ~transit_mask

    if np.sum(oot_mask) < 2 or np.sum(transit_mask) < 1:
        logger.warning("Insufficient data for SNR computation.")
        return 0.0

    oot_flux = flux[oot_mask]
    transit_flux = flux[transit_mask]

    baseline = np.median(oot_flux)
    transit_depth = abs(baseline - np.median(transit_flux))

    oot_std = np.std(oot_flux)
    if oot_std == 0:
        return 0.0

    snr = transit_depth / oot_std
    return float(snr)


def extract_all_features(
    time: np.ndarray,
    flux: np.ndarray,
    period_min: float = BLS_PERIOD_MIN,
    period_max: float = BLS_PERIOD_MAX,
    n_periods: int = BLS_PERIOD_NPOINTS,
) -> Dict:
    logger.info("Starting full feature extraction pipeline.")

    try:
        bls_results = run_bls_search(
            time, flux,
            period_min=period_min,
            period_max=period_max,
            n_periods=n_periods,
        )

        best_period = bls_results["best_period"]
        best_t0 = bls_results["best_t0"]

        phase, folded_flux = fold_lightcurve(time, flux, best_period, best_t0)

        transit_params = extract_transit_params(time, flux, best_period, best_t0)

        features = {**bls_results, **transit_params}
        features["phase"] = phase
        features["folded_flux"] = folded_flux

        logger.info(
            "Feature extraction complete — period=%.5f d, depth=%.6f, snr=%.2f",
            features["best_period"],
            features["depth"],
            features["snr"],
        )

        return features

    except Exception as exc:
        logger.error("Feature extraction pipeline failed: %s", exc)
        raise
