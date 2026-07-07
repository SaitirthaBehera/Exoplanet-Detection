"""
Typed configuration loader for the ExoVision AI preprocessing pipeline.

This module reads ``config/preprocessing.yaml`` and exposes it as a set of
small, typed dataclasses (:class:`PreprocessingConfig` and its nested
sections). Downstream preprocessing modules should depend on these
dataclasses rather than reading YAML/dict data directly — this keeps every
configurable knob discoverable, type-checked, and centrally documented.

This module is purely additive: it does not alter ``src/config.py`` or any
existing constant used by the models, classifier, or dashboard.

Example
-------
>>> from src.preprocessing_config import load_preprocessing_config
>>> cfg = load_preprocessing_config()
>>> cfg.outliers.method
'sigma_clip'
>>> cfg.windowing.window_size
2001
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# Default location of the YAML config, relative to the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "preprocessing.yaml")


# ---------------------------------------------------------------------------
# Nested configuration sections
# ---------------------------------------------------------------------------

@dataclass
class MissingValuesConfig:
    """Configuration for Section 3 — Missing Value Handling."""
    strategy: str = "linear"  # linear | cubic | nearest | ffill | bfill


@dataclass
class OutlierConfig:
    """Configuration for Section 4 — Outlier Detection."""
    method: str = "sigma_clip"  # sigma_clip | mad | percentile
    sigma_threshold: float = 5.0
    mad_threshold: float = 3.5
    percentile_lower: float = 0.5
    percentile_upper: float = 99.5


@dataclass
class NormalizationConfig:
    """Configuration for Section 5 — Normalization."""
    method: str = "median"  # minmax | zscore | median | robust


@dataclass
class DetrendingConfig:
    """Configuration for Section 6 — Trend Removal."""
    enabled: bool = False
    method: str = "savgol"  # polynomial | savgol | running_median
    polynomial_degree: int = 3
    savgol_window: int = 401
    savgol_polyorder: int = 3
    running_median_window: int = 101


@dataclass
class WindowingConfig:
    """Configuration for Section 7 — Window Segmentation."""
    window_size: int = 2001
    overlap: float = 0.75  # fraction in [0, 1)
    padding_mode: str = "edge"

    @property
    def stride(self) -> int:
        """Compute the sliding-window stride implied by ``overlap``."""
        stride = int(round(self.window_size * (1.0 - self.overlap)))
        return max(stride, 1)


@dataclass
class AugmentationStepConfig:
    """A single augmentation technique's configuration."""
    probability: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    """Configuration for Section 8 — Data Augmentation (train-time only)."""
    enabled: bool = False
    gaussian_noise: AugmentationStepConfig = field(
        default_factory=lambda: AugmentationStepConfig(
            0.5, {"sigma_range": (0.0005, 0.003)}
        )
    )
    time_shift: AugmentationStepConfig = field(
        default_factory=lambda: AugmentationStepConfig(
            0.3, {"max_shift_fraction": 0.02}
        )
    )
    flux_scaling: AugmentationStepConfig = field(
        default_factory=lambda: AugmentationStepConfig(
            0.3, {"scale_range": (0.98, 1.02)}
        )
    )
    random_masking: AugmentationStepConfig = field(
        default_factory=lambda: AugmentationStepConfig(
            0.2, {"max_mask_fraction": 0.05}
        )
    )
    baseline_drift: AugmentationStepConfig = field(
        default_factory=lambda: AugmentationStepConfig(
            0.3, {"amplitude_range": (0.0002, 0.002)}
        )
    )


@dataclass
class OutputConfig:
    """Configuration for Section 12 — Outputs."""
    processed_dir: str = "data/processed"
    save_windows: bool = True


@dataclass
class PreprocessingConfig:
    """
    Root configuration object for the preprocessing pipeline.

    Attributes
    ----------
    random_seed:
        Global seed used for any stochastic step (augmentation, synthetic
        data generation) to keep runs reproducible.
    missing_values, outliers, normalization, detrending, windowing,
    augmentation, output:
        Nested section configs — see each dataclass's docstring.
    noise_level:
        Baseline noise-sigma used where a default is needed (e.g. synthetic
        data generation, noise-based augmentation fallback).
    """
    random_seed: int = 42
    missing_values: MissingValuesConfig = field(default_factory=MissingValuesConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    detrending: DetrendingConfig = field(default_factory=DetrendingConfig)
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    noise_level: float = 0.001


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _build_augmentation_config(raw: Dict[str, Any]) -> AugmentationConfig:
    """Build an :class:`AugmentationConfig` from a raw YAML dict."""
    def _step(name: str, default: AugmentationStepConfig) -> AugmentationStepConfig:
        raw_step = raw.get(name)
        if not isinstance(raw_step, dict):
            return default
        probability = float(raw_step.get("probability", default.probability))
        params = {k: v for k, v in raw_step.items() if k != "probability"}
        # Normalize any [lo, hi] YAML lists to tuples for downstream code.
        params = {
            k: (tuple(v) if isinstance(v, list) else v) for k, v in params.items()
        }
        return AugmentationStepConfig(probability=probability, params=params or default.params)

    defaults = AugmentationConfig()
    return AugmentationConfig(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        gaussian_noise=_step("gaussian_noise", defaults.gaussian_noise),
        time_shift=_step("time_shift", defaults.time_shift),
        flux_scaling=_step("flux_scaling", defaults.flux_scaling),
        random_masking=_step("random_masking", defaults.random_masking),
        baseline_drift=_step("baseline_drift", defaults.baseline_drift),
    )


def load_preprocessing_config(
    config_path: Optional[str] = None,
) -> PreprocessingConfig:
    """
    Load the preprocessing configuration from a YAML file.

    Parameters
    ----------
    config_path:
        Path to the YAML config file. Defaults to
        ``config/preprocessing.yaml`` at the project root.

    Returns
    -------
    PreprocessingConfig
        A fully populated, typed configuration object. Any key missing from
        the YAML file falls back to the dataclass defaults, so a partial or
        even empty file is safe to use.

    Raises
    ------
    yaml.YAMLError
        If the file exists but contains invalid YAML.

    Notes
    -----
    This function never raises on a *missing* file — it logs a warning and
    returns default values, so the rest of the pipeline keeps working
    (e.g. in fresh clones before the file is customized).
    """
    path = config_path or DEFAULT_CONFIG_PATH

    if not os.path.isfile(path):
        logger.warning(
            "Preprocessing config not found at '%s'. Using built-in defaults.",
            path,
        )
        return PreprocessingConfig()

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}
    except yaml.YAMLError:
        logger.exception("Failed to parse YAML config at '%s'.", path)
        raise

    defaults = PreprocessingConfig()

    cfg = PreprocessingConfig(
        random_seed=int(raw.get("random_seed", defaults.random_seed)),
        missing_values=MissingValuesConfig(
            strategy=raw.get("missing_values", {}).get(
                "strategy", defaults.missing_values.strategy
            )
        ),
        outliers=OutlierConfig(
            method=raw.get("outliers", {}).get("method", defaults.outliers.method),
            sigma_threshold=float(
                raw.get("outliers", {}).get(
                    "sigma_threshold", defaults.outliers.sigma_threshold
                )
            ),
            mad_threshold=float(
                raw.get("outliers", {}).get(
                    "mad_threshold", defaults.outliers.mad_threshold
                )
            ),
            percentile_lower=float(
                raw.get("outliers", {}).get(
                    "percentile_lower", defaults.outliers.percentile_lower
                )
            ),
            percentile_upper=float(
                raw.get("outliers", {}).get(
                    "percentile_upper", defaults.outliers.percentile_upper
                )
            ),
        ),
        normalization=NormalizationConfig(
            method=raw.get("normalization", {}).get(
                "method", defaults.normalization.method
            )
        ),
        detrending=DetrendingConfig(
            enabled=bool(
                raw.get("detrending", {}).get("enabled", defaults.detrending.enabled)
            ),
            method=raw.get("detrending", {}).get(
                "method", defaults.detrending.method
            ),
            polynomial_degree=int(
                raw.get("detrending", {}).get(
                    "polynomial_degree", defaults.detrending.polynomial_degree
                )
            ),
            savgol_window=int(
                raw.get("detrending", {}).get(
                    "savgol_window", defaults.detrending.savgol_window
                )
            ),
            savgol_polyorder=int(
                raw.get("detrending", {}).get(
                    "savgol_polyorder", defaults.detrending.savgol_polyorder
                )
            ),
            running_median_window=int(
                raw.get("detrending", {}).get(
                    "running_median_window",
                    defaults.detrending.running_median_window,
                )
            ),
        ),
        windowing=WindowingConfig(
            window_size=int(
                raw.get("windowing", {}).get(
                    "window_size", defaults.windowing.window_size
                )
            ),
            overlap=float(
                raw.get("windowing", {}).get("overlap", defaults.windowing.overlap)
            ),
            padding_mode=raw.get("windowing", {}).get(
                "padding_mode", defaults.windowing.padding_mode
            ),
        ),
        augmentation=_build_augmentation_config(raw.get("augmentation", {})),
        output=OutputConfig(
            processed_dir=raw.get("output", {}).get(
                "processed_dir", defaults.output.processed_dir
            ),
            save_windows=bool(
                raw.get("output", {}).get(
                    "save_windows", defaults.output.save_windows
                )
            ),
        ),
        noise_level=float(raw.get("noise_level", defaults.noise_level)),
    )

    logger.info("Loaded preprocessing config from '%s'.", path)
    return cfg
