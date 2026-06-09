"""Error distribution visualization utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_error_distribution(
    error_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot error distribution by model.

    Args:
        error_df: DataFrame with model, error_type, and count columns.
        output_path: Output figure path.
    """
    pivot_df = error_df.pivot(
        index="model",
        columns="error_type",
        values="count",
    ).fillna(0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot_df.plot(
        kind="bar",
        figsize=(8, 5),
    )

    plt.title("Error Distribution by Model")
    plt.xlabel("Model")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_error_rate(
    error_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot error rates by model.

    Args:
        error_df: DataFrame with model, error_type, and count columns.
        output_path: Output figure path.
    """
    pivot_df = error_df.pivot(
        index="model",
        columns="error_type",
        values="count",
    ).fillna(0)

    rate_df = pivot_df.div(
        pivot_df.sum(axis=1),
        axis=0,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rate_df.plot(
        kind="bar",
        figsize=(8, 5),
    )

    plt.title("Error Rate by Model")
    plt.xlabel("Model")
    plt.ylabel("Rate")
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
