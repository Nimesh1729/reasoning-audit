"""Model comparison utilities."""

import pandas as pd


def build_comparison_table(
    model_results: list[dict[str, float | str]],
) -> pd.DataFrame:
    """Build model comparison table.

    Args:
        model_results: Metrics for each model.

    Returns:
        Comparison DataFrame.
    """
    return pd.DataFrame(model_results)
