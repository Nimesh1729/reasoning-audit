"""Utilities for expanding benchmark questions into prompt cases."""

import pandas as pd

from src.prompts.prompt_generator import (
    generate_clean_prompt,
    generate_helpful_prompt,
    generate_misleading_prompt,
)


def build_prompt_cases(
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Expand benchmark questions into clean/helpful/misleading prompt cases.

    Args:
        benchmark: Benchmark DataFrame with question, answer, helpful_hint,
            and misleading_hint columns.

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
