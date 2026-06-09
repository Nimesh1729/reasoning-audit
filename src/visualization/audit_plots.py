"""Visualization utilities for reasoning-audit metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_overall_accuracy(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot overall accuracy.

    Args:
        comparison_df: Model comparison DataFrame.
        output_path: Output figure path.
    """
    plt.figure(figsize=(8, 5))

    plt.bar(
        comparison_df["model"],
        comparison_df["overall_accuracy"],
    )

    plt.title("Overall Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_misleading_accuracy(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot misleading-prompt accuracy.

    Args:
        comparison_df: Model comparison DataFrame.
        output_path: Output figure path.
    """
    plt.figure(figsize=(8, 5))

    plt.bar(
        comparison_df["model"],
        comparison_df["misleading_accuracy"],
    )

    plt.title("Misleading Prompt Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_misleading_hint_drop(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot misleading-hint drop.

    Args:
        comparison_df: Model comparison DataFrame.
        output_path: Output figure path.
    """
    plt.figure(figsize=(8, 5))

    plt.bar(
        comparison_df["model"],
        comparison_df["misleading_hint_drop"],
    )

    plt.title("Misleading Hint Drop")
    plt.ylabel("Accuracy Drop")
    plt.ylim(0.0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
