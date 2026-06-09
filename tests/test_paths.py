"""Tests for path utilities."""

from pathlib import Path

from src.utils.paths import (
    get_data_dir,
    get_logs_dir,
    get_outputs_dir,
    get_samples_dir,
)


def test_project_paths() -> None:
    """Test project path utilities."""
    assert get_data_dir() == Path("data")
    assert get_outputs_dir() == Path("outputs")
    assert get_logs_dir() == Path("logs")
    assert get_samples_dir() == Path("data") / "samples"
