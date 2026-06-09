"""Plot reasoning-audit error distributions."""

from pathlib import Path

import pandas as pd

from src.visualization.error_plots import (
    plot_error_distribution,
    plot_error_rate,
)


def main() -> None:
    """Generate error distribution plot."""
    hint_sensitive_path = Path("outputs/hint_sensitive/error_distribution.csv")
    semi_robust_path = Path("outputs/semi_robust/error_distribution.csv")

    hint_sensitive_df = pd.read_csv(hint_sensitive_path)
    hint_sensitive_df["model"] = "HintSensitive"

    semi_robust_df = pd.read_csv(semi_robust_path)
    semi_robust_df["model"] = "SemiRobust"

    error_df = pd.concat(
        [
            hint_sensitive_df,
            semi_robust_df,
        ],
        ignore_index=True,
    )

    output_path = Path("outputs/figures/error_distribution.png")

    plot_error_distribution(
        error_df=error_df,
        output_path=output_path,
    )

    print(f"Saved error distribution plot to: {output_path}")

    error_rate_path = Path("outputs/figures/error_rate.png")

    plot_error_rate(
        error_df=error_df,
        output_path=error_rate_path,
    )

    print(f"Saved error rate plot to: {error_rate_path}")


if __name__ == "__main__":
    main()
