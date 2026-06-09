"""Split benchmark into train and test sets."""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger


def split_by_domain(
    benchmark: pd.DataFrame,
    test_counts: dict[str, int],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split benchmark into stratified train/test sets.

    Args:
        benchmark: Full benchmark DataFrame.
        test_counts: Number of test examples per domain.
        seed: Random seed.

    Returns:
        Train and test DataFrames.
    """
    test_parts = []
    train_parts = []

    for domain, domain_df in benchmark.groupby("domain"):
        shuffled_df = domain_df.sample(
            frac=1.0,
            random_state=seed,
        )

        num_test = test_counts.get(domain, 0)

        test_parts.append(
            shuffled_df.head(num_test),
        )
        train_parts.append(
            shuffled_df.iloc[num_test:],
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True,
    )

    return train_df, test_df


def main() -> None:
    """Split benchmark and save train/test CSVs."""
    logger = get_logger(__name__)

    input_path = Path("data/benchmark/questions_v1_1.csv")
    train_path = Path("data/benchmark/train.csv")
    test_path = Path("data/benchmark/test.csv")

    benchmark = pd.read_csv(input_path)

    test_counts = {
        "astronomy": 3,
        "logic": 3,
        "physics": 2,
        "arithmetic": 2,
    }

    train_df, test_df = split_by_domain(
        benchmark=benchmark,
        test_counts=test_counts,
        seed=42,
    )

    train_df.to_csv(
        train_path,
        index=False,
    )
    test_df.to_csv(
        test_path,
        index=False,
    )

    logger.info("Full benchmark size: %d", len(benchmark))
    logger.info("Train size: %d", len(train_df))
    logger.info("Test size: %d", len(test_df))
    logger.info("Train domain counts:\n%s", train_df["domain"].value_counts())
    logger.info("Test domain counts:\n%s", test_df["domain"].value_counts())
    logger.info("Saved train split to %s", train_path)
    logger.info("Saved test split to %s", test_path)


if __name__ == "__main__":
    main()
