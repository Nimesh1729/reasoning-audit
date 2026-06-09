"""Compare reasoning-audit models."""

from pathlib import Path

import pandas as pd

from src.analysis.model_comparison import build_comparison_table


def main() -> None:
    """Run model comparison."""
    comparison_df = build_comparison_table(
        [
            {
                "model": "RuleBased",
                "overall_accuracy": 1.0,
                "clean_accuracy": 1.0,
                "helpful_accuracy": 1.0,
                "misleading_accuracy": 1.0,
                "helpful_hint_gain": 0.0,
                "misleading_hint_drop": 0.0,
            },
            {
                "model": "SemiRobust",
                "overall_accuracy": 0.833333,
                "clean_accuracy": 1.0,
                "helpful_accuracy": 1.0,
                "misleading_accuracy": 0.5,
                "helpful_hint_gain": 0.0,
                "misleading_hint_drop": 0.5,
            },
            {
                "model": "HintSensitive",
                "overall_accuracy": 0.666667,
                "clean_accuracy": 1.0,
                "helpful_accuracy": 1.0,
                "misleading_accuracy": 0.0,
                "helpful_hint_gain": 0.0,
                "misleading_hint_drop": 1.0,
            },
        ]
    )

    output_dir = Path("outputs/comparison")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "model_comparison.csv"

    comparison_df.to_csv(
        output_path,
        index=False,
    )

    print(comparison_df)
    print(f"\nSaved comparison to: {output_path}")


if __name__ == "__main__":
    main()
