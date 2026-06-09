"""Training dataset utilities for FLAN-T5 fine-tuning."""

from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerBase


def load_training_dataframe(
    csv_path: str | Path,
) -> pd.DataFrame:
    """Load training CSV.

    Args:
        csv_path: Path to training CSV.

    Returns:
        Training DataFrame.
    """
    return pd.read_csv(csv_path)


def build_instruction_dataframe(
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert benchmark rows into instruction-tuning rows.

    Args:
        benchmark_df: Benchmark DataFrame.

    Returns:
        DataFrame with input_text and target_text columns.
    """
    rows = []

    is_mcq = {
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "answer_key",
    }.issubset(benchmark_df.columns)

    for _, row in benchmark_df.iterrows():
        if is_mcq:
            prompt = (
                "Answer with only the option letter: "
                "A, B, C, or D.\n\n"
                f"Question:\n{row['question']}\n\n"
                "Options:\n"
                f"A. {row['option_a']}\n"
                f"B. {row['option_b']}\n"
                f"C. {row['option_c']}\n"
                f"D. {row['option_d']}"
            )

            target = row["answer_key"]

        else:
            prompt = f"Answer with only the final answer.\n\n{row['question']}"

            target = row["answer"]

        rows.append(
            {
                "input_text": prompt,
                "target_text": target,
            }
        )

    return pd.DataFrame(rows)


def build_hf_dataset(
    instruction_df: pd.DataFrame,
) -> Dataset:
    """Convert instruction DataFrame to Hugging Face Dataset.

    Args:
        instruction_df: DataFrame with input_text and target_text columns.

    Returns:
        Hugging Face Dataset.
    """
    return Dataset.from_pandas(
        instruction_df,
        preserve_index=False,
    )


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int = 128,
    max_target_length: int = 32,
) -> Dataset:
    """Tokenize instruction dataset.

    Args:
        dataset: Hugging Face Dataset.
        tokenizer: Tokenizer.
        max_input_length: Maximum input token length.
        max_target_length: Maximum target token length.

    Returns:
        Tokenized Hugging Face Dataset.
    """

    def tokenize_batch(
        batch: dict[str, list[str]],
    ) -> dict[str, Any]:
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_input_length,
            truncation=True,
        )

        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return dataset.map(
        tokenize_batch,
        batched=True,
    )
