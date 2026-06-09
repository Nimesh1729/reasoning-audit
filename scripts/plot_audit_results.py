"""Generate reasoning-audit visualizations."""

from pathlib import Path

import pandas as pd

from src.visualization.audit_plots import (
    plot_misleading_accuracy,
    plot_misleading_hint_drop,
    plot_overall_accuracy,
)


def main() -> None:
    """Generate audit figures."""
    comparison_csv = Path("outputs") / "comparison" / "model_comparison.csv"

    comparison_df = pd.read_csv(
        comparison_csv,
    )

    figures_dir = Path("outputs") / "figures"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_overall_accuracy(
        comparison_df=comparison_df,
        output_path=(figures_dir / "overall_accuracy.png"),
    )

    plot_misleading_accuracy(
        comparison_df=comparison_df,
        output_path=(figures_dir / "misleading_accuracy.png"),
    )

    plot_misleading_hint_drop(
        comparison_df=comparison_df,
        output_path=(figures_dir / "misleading_hint_drop.png"),
    )

    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
