"""Benchmark loading utilities."""

from pathlib import Path

import pandas as pd


def load_benchmark(
    csv_path: str | Path,
) -> pd.DataFrame:
    """Load reasoning benchmark from CSV.

    Args:
        csv_path: Path to benchmark CSV.

    Returns:
        Benchmark DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    benchmark = pd.read_csv(csv_path)

    required_columns = {
        "id",
        "domain",
        "difficulty",
        "question",
        "answer",
        "helpful_hint",
        "misleading_hint",
    }

    if not required_columns.issubset(benchmark.columns):
        raise ValueError(
            "CSV must contain id, domain, difficulty, question, "
            "answer, helpful_hint, and misleading_hint columns."
        )

    return benchmark
