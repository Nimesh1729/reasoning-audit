"""Tests for benchmark prompt-case expansion."""

import pandas as pd

from src.evaluation.benchmark_dataset import build_prompt_cases


def test_build_prompt_cases() -> None:
    """Test benchmark expansion into three prompt types."""
    benchmark = pd.DataFrame(
        {
            "id": [1],
            "domain": ["astronomy"],
            "difficulty": ["easy"],
            "question": ["What type of object is the Sun?"],
            "answer": ["star"],
            "helpful_hint": ["It generates energy through nuclear fusion."],
            "misleading_hint": ["It is the largest planet."],
        }
    )

    prompt_cases = build_prompt_cases(benchmark)

    assert len(prompt_cases) == 3
    assert set(prompt_cases["prompt_type"]) == {
        "clean",
        "helpful",
        "misleading",
    }
    assert prompt_cases.iloc[0]["question_id"] == 1
    assert "nuclear fusion" in prompt_cases.iloc[1]["prompt"]
    assert "largest planet" in prompt_cases.iloc[2]["prompt"]
