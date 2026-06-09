"""Split benchmark into train and test sets."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger


def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = ArgumentParser(
        description="Split benchmark into train and test CSV files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmark/questions_v2_mcq.csv"),
        help="Path to full benchmark CSV.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/benchmark/train.csv"),
        help="Path where train split will be saved.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("data/benchmark/test.csv"),
        help="Path where test split will be saved.",
    )

    return parser.parse_args()


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
    args = parse_args()

    benchmark = pd.read_csv(args.input)

    test_counts = {
        "astronomy": 5,
        "logic": 5,
        "physics": 5,
        "arithmetic": 5,
    }

    train_df, test_df = split_by_domain(
        benchmark=benchmark,
        test_counts=test_counts,
        seed=42,
    )

    train_df.to_csv(
        args.train_output,
        index=False,
    )
    test_df.to_csv(
        args.test_output,
        index=False,
    )

    logger.info("Input benchmark: %s", args.input)
    logger.info("Full benchmark size: %d", len(benchmark))
    logger.info("Train size: %d", len(train_df))
    logger.info("Test size: %d", len(test_df))
    logger.info("Train domain counts:\n%s", train_df["domain"].value_counts())
    logger.info("Test domain counts:\n%s", test_df["domain"].value_counts())
    logger.info("Saved train split to %s", args.train_output)
    logger.info("Saved test split to %s", args.test_output)


if __name__ == "__main__":
    main()
