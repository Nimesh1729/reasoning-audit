"""Tests for training dataset utilities."""

import pandas as pd

from src.training.dataset import (
    build_hf_dataset,
    build_instruction_dataframe,
)


def test_build_instruction_dataframe() -> None:
    """Instruction dataframe should be created correctly."""
    benchmark_df = pd.DataFrame(
        {
            "question": [
                "What type of object is the Sun?",
            ],
            "answer": [
                "star",
            ],
        }
    )

    instruction_df = build_instruction_dataframe(
        benchmark_df,
    )

    assert len(instruction_df) == 1

    assert instruction_df.iloc[0]["input_text"] == (
        "Answer with only the final answer.\n\nWhat type of object is the Sun?"
    )

    assert instruction_df.iloc[0]["target_text"] == "star"


def test_build_hf_dataset() -> None:
    """HF dataset should have correct size."""
    instruction_df = pd.DataFrame(
        {
            "input_text": [
                "Question 1",
                "Question 2",
            ],
            "target_text": [
                "Answer 1",
                "Answer 2",
            ],
        }
    )

    dataset = build_hf_dataset(
        instruction_df,
    )

    assert len(dataset) == 2

    assert dataset[0]["input_text"] == "Question 1"

    assert dataset[0]["target_text"] == "Answer 1"


def test_tokenize_dataset() -> None:
    """Tokenized dataset should include model inputs and labels."""
    from transformers import AutoTokenizer

    from src.training.dataset import tokenize_dataset

    instruction_df = pd.DataFrame(
        {
            "input_text": [
                "Answer with only the final answer.\n\nWhat force keeps planets in orbit?",
            ],
            "target_text": [
                "gravity",
            ],
        }
    )

    dataset = build_hf_dataset(
        instruction_df,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-small",
    )

    tokenized_dataset = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
    )

    assert len(tokenized_dataset) == 1
    assert "input_ids" in tokenized_dataset.column_names
    assert "attention_mask" in tokenized_dataset.column_names
    assert "labels" in tokenized_dataset.column_names
