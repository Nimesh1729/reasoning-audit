"""Tests for model comparison utilities."""

from src.analysis.model_comparison import build_comparison_table


def test_build_comparison_table() -> None:
    """Test comparison table creation."""
    model_results = [
        {
            "model": "RuleBased",
            "overall_accuracy": 1.0,
        },
        {
            "model": "HintSensitive",
            "overall_accuracy": 0.667,
        },
    ]

    comparison_df = build_comparison_table(
        model_results,
    )

    assert len(comparison_df) == 2
    assert "model" in comparison_df.columns
