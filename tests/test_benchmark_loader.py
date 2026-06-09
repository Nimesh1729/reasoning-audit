"""Tests for benchmark loading."""

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.benchmark_loader import load_benchmark


def test_load_benchmark(
    tmp_path: Path,
) -> None:
    """Test loading a valid benchmark CSV."""
    benchmark_file = tmp_path / "questions.csv"

    pd.DataFrame(
        {
            "id": [1],
            "domain": ["astronomy"],
            "difficulty": ["easy"],
            "question": ["What type of object is the Sun?"],
            "answer": ["star"],
            "helpful_hint": ["It generates energy through nuclear fusion."],
            "misleading_hint": ["It is the largest planet."],
        }
    ).to_csv(
        benchmark_file,
        index=False,
    )

    benchmark = load_benchmark(benchmark_file)

    assert len(benchmark) == 1
    assert benchmark.iloc[0]["domain"] == "astronomy"
    assert benchmark.iloc[0]["answer"] == "star"


def test_missing_columns(
    tmp_path: Path,
) -> None:
    """Test benchmark CSV with missing required columns."""
    benchmark_file = tmp_path / "questions.csv"

    pd.DataFrame(
        {
            "question": ["What type of object is the Sun?"],
            "answer": ["star"],
        }
    ).to_csv(
        benchmark_file,
        index=False,
    )

    with pytest.raises(ValueError):
        load_benchmark(benchmark_file)
