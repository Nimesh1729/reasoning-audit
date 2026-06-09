"""Path utilities."""

from pathlib import Path


def get_data_dir() -> Path:
    """Return data directory."""
    return Path("data")


def get_outputs_dir() -> Path:
    """Return outputs directory."""
    return Path("outputs")


def get_logs_dir() -> Path:
    """Return logs directory."""
    return Path("logs")


def get_samples_dir() -> Path:
    """Return sample data directory."""
    return get_data_dir() / "samples"
