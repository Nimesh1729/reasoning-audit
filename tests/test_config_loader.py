"""Tests for config loading."""

from pathlib import Path

import pytest

from src.utils.config_loader import load_config


def test_load_config(
    tmp_path: Path,
) -> None:
    """Test loading a valid config."""
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        "seed: 42\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config["seed"] == 42


def test_missing_config() -> None:
    """Test missing config file."""
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")
