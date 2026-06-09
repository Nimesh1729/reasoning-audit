"""Utilities for expanding benchmark questions into prompt cases."""

import pandas as pd

from src.prompts.prompt_generator import (
    generate_clean_mcq_prompt,
    generate_clean_prompt,
    generate_helpful_mcq_prompt,
    generate_helpful_prompt,
    generate_misleading_mcq_prompt,
    generate_misleading_prompt,
)

MCQ_COLUMNS = {
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "answer_key",
}


def has_mcq_columns(
    benchmark: pd.DataFrame,
) -> bool:
    """Check whether benchmark uses multiple-choice format.

    Args:
        benchmark: Benchmark DataFrame.

    Returns:
        Whether benchmark contains MCQ columns.
    """
    return MCQ_COLUMNS.issubset(benchmark.columns)


def build_prompt_cases(
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Expand benchmark questions into prompt cases.

    Args:
        benchmark: Benchmark DataFrame.

    Returns:
        DataFrame containing one row per prompt case.
    """
    if has_mcq_columns(benchmark):
        return build_mcq_prompt_cases(benchmark)

    return build_freeform_prompt_cases(benchmark)


def build_freeform_prompt_cases(
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Expand free-form benchmark questions into prompt cases.

    Args:
        benchmark: Free-form benchmark DataFrame.

    Returns:
        DataFrame containing one row per prompt case.
    """
    rows = []

    for _, row in benchmark.iterrows():
        base_metadata = {
            "question_id": row["id"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
            "answer": row["answer"],
        }

        rows.append(
            {
                **base_metadata,
                "prompt_type": "clean",
                "prompt": generate_clean_prompt(row["question"]),
            }
        )

        rows.append(
            {
                **base_metadata,
                "prompt_type": "helpful",
                "prompt": generate_helpful_prompt(
                    question=row["question"],
                    helpful_hint=row["helpful_hint"],
                ),
            }
        )

        rows.append(
            {
                **base_metadata,
                "prompt_type": "misleading",
                "prompt": generate_misleading_prompt(
                    question=row["question"],
                    misleading_hint=row["misleading_hint"],
                ),
            }
        )

    return pd.DataFrame(rows)


def build_mcq_prompt_cases(
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Expand MCQ benchmark questions into prompt cases.

    Args:
        benchmark: MCQ benchmark DataFrame.

    Returns:
        DataFrame containing one row per prompt case.
    """
    rows = []

    for _, row in benchmark.iterrows():
        base_metadata = {
            "question_id": row["id"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
            "answer": row["answer_key"],
        }

        rows.append(
            {
                **base_metadata,
                "prompt_type": "clean",
                "prompt": generate_clean_mcq_prompt(
                    question=row["question"],
                    option_a=row["option_a"],
                    option_b=row["option_b"],
                    option_c=row["option_c"],
                    option_d=row["option_d"],
                ),
            }
        )

        rows.append(
            {
                **base_metadata,
                "prompt_type": "helpful",
                "prompt": generate_helpful_mcq_prompt(
                    question=row["question"],
                    option_a=row["option_a"],
                    option_b=row["option_b"],
                    option_c=row["option_c"],
                    option_d=row["option_d"],
                    helpful_hint=row["helpful_hint"],
                ),
            }
        )

        rows.append(
            {
                **base_metadata,
                "prompt_type": "misleading",
                "prompt": generate_misleading_mcq_prompt(
                    question=row["question"],
                    option_a=row["option_a"],
                    option_b=row["option_b"],
                    option_c=row["option_c"],
                    option_d=row["option_d"],
                    misleading_hint=row["misleading_hint"],
                ),
            }
        )

    return pd.DataFrame(rows)
