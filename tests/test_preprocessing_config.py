"""Unit tests for src.preprocessing_config (Section 14: Configuration Loading)."""

import os
import textwrap

import pytest

from src.preprocessing_config import (
    PreprocessingConfig,
    load_preprocessing_config,
)


def test_load_default_config_file():
    """The bundled config/preprocessing.yaml should load without error."""
    cfg = load_preprocessing_config()
    assert isinstance(cfg, PreprocessingConfig)
    assert cfg.windowing.window_size == 2001
    assert cfg.outliers.method == "sigma_clip"


def test_missing_file_falls_back_to_defaults(tmp_path):
    """A nonexistent path should log a warning and return built-in defaults, not raise."""
    cfg = load_preprocessing_config(str(tmp_path / "does_not_exist.yaml"))
    assert isinstance(cfg, PreprocessingConfig)
    assert cfg.random_seed == 42
    assert cfg.normalization.method == "median"


def test_partial_config_uses_defaults_for_missing_keys(tmp_path):
    """A YAML file specifying only some keys should fall back to defaults for the rest."""
    partial = tmp_path / "partial.yaml"
    partial.write_text(textwrap.dedent("""
        random_seed: 123
        outliers:
          method: mad
    """))
    cfg = load_preprocessing_config(str(partial))
    assert cfg.random_seed == 123
    assert cfg.outliers.method == "mad"
    # Untouched sections keep their defaults.
    assert cfg.normalization.method == "median"
    assert cfg.windowing.window_size == 2001


def test_empty_file_uses_all_defaults(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    cfg = load_preprocessing_config(str(empty))
    assert cfg == PreprocessingConfig()


def test_invalid_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("outliers: [this is: not valid: yaml")
    with pytest.raises(Exception):
        load_preprocessing_config(str(bad))


def test_windowing_stride_property():
    cfg = PreprocessingConfig()
    cfg.windowing.window_size = 2001
    cfg.windowing.overlap = 0.75
    assert cfg.windowing.stride == 500


def test_augmentation_list_params_become_tuples(tmp_path):
    """YAML [lo, hi] lists for augmentation params should load as tuples."""
    cfg_file = tmp_path / "aug.yaml"
    cfg_file.write_text(textwrap.dedent("""
        augmentation:
          enabled: true
          gaussian_noise:
            probability: 0.9
            sigma_range: [0.001, 0.005]
    """))
    cfg = load_preprocessing_config(str(cfg_file))
    assert cfg.augmentation.enabled is True
    assert cfg.augmentation.gaussian_noise.probability == 0.9
    assert cfg.augmentation.gaussian_noise.params["sigma_range"] == (0.001, 0.005)
