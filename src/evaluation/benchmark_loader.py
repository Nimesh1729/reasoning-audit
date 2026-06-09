"""Benchmark loading utilities."""

from pathlib import Path

import pandas as pd

FREEFORM_COLUMNS = {
    "id",
    "domain",
    "difficulty",
    "question",
    "answer",
    "helpful_hint",
    "misleading_hint",
}

MCQ_COLUMNS = {
    "id",
    "domain",
    "difficulty",
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "answer_key",
    "helpful_hint",
    "misleading_hint",
}


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
    columns = set(benchmark.columns)

    is_freeform = FREEFORM_COLUMNS.issubset(columns)
    is_mcq = MCQ_COLUMNS.issubset(columns)

    if not (is_freeform or is_mcq):
        raise ValueError(
            "CSV must be either a free-form benchmark with columns "
            "id, domain, difficulty, question, answer, helpful_hint, "
            "and misleading_hint, or an MCQ benchmark with columns "
            "id, domain, difficulty, question, option_a, option_b, "
            "option_c, option_d, answer_key, helpful_hint, "
            "and misleading_hint."
        )

    return benchmark
