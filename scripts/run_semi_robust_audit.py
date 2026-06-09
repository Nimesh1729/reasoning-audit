"""Run a hint-sensitive reasoning-audit experiment."""

from pathlib import Path

import pandas as pd

from src.analysis.error_analysis import compute_error_distribution
from src.evaluation.benchmark_dataset import build_prompt_cases
from src.evaluation.benchmark_loader import load_benchmark
from src.evaluation.evaluator import evaluate_prompt_cases
from src.evaluation.metrics import (
    compute_accuracy,
    compute_accuracy_by_domain,
    compute_accuracy_by_prompt_type,
    compute_helpful_hint_gain,
    compute_misleading_hint_drop,
)
from src.models.semi_robust_model import SemiRobustModel
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


def main() -> None:
    """Run hint-sensitive audit pipeline."""
    logger = get_logger(__name__)
    args = parse_args()

    config = load_config(args.config)
    benchmark_csv = config["data"]["benchmark_csv"]

    benchmark = load_benchmark(benchmark_csv)
    prompt_cases = build_prompt_cases(benchmark)

    model = SemiRobustModel()

    results = evaluate_prompt_cases(
        prompt_cases=prompt_cases,
        model=model,
    )

    overall_accuracy = compute_accuracy(results)
    accuracy_by_prompt_type = compute_accuracy_by_prompt_type(results)
    accuracy_by_domain = compute_accuracy_by_domain(results)

    helpful_hint_gain = compute_helpful_hint_gain(
        accuracy_by_prompt_type,
    )
    misleading_hint_drop = compute_misleading_hint_drop(
        accuracy_by_prompt_type,
    )

    results_df = pd.DataFrame(
        [
            {
                "question_id": result.question_id,
                "domain": result.domain,
                "prompt_type": result.prompt_type,
                "ground_truth": result.ground_truth,
                "prediction": result.prediction,
                "correct": result.correct,
            }
            for result in results
        ]
    )

    error_distribution = compute_error_distribution(results)

    error_distribution_df = pd.DataFrame(
        [
            {
                "error_type": error_type,
                "count": count,
            }
            for error_type, count in error_distribution.items()
        ]
    )

    output_dir = Path("outputs/semi_robust")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.csv"
    metrics_path = output_dir / "metrics.csv"
    error_distribution_path = output_dir / "error_distribution.csv"

    logger.info("Error distribution: %s", error_distribution)
    logger.info("Saved error distribution to %s", error_distribution_path)

    results_df.to_csv(results_path, index=False)

    metrics_df = pd.DataFrame(
        [
            {
                "overall_accuracy": overall_accuracy,
                "clean_accuracy": accuracy_by_prompt_type["clean"],
                "helpful_accuracy": accuracy_by_prompt_type["helpful"],
                "misleading_accuracy": accuracy_by_prompt_type["misleading"],
                "helpful_hint_gain": helpful_hint_gain,
                "misleading_hint_drop": misleading_hint_drop,
                "astronomy_accuracy": accuracy_by_domain["astronomy"],
                "logic_accuracy": accuracy_by_domain["logic"],
            }
        ]
    )

    metrics_df.to_csv(metrics_path, index=False)
    error_distribution_df.to_csv(
        error_distribution_path,
        index=False,
    )

    logger.info("Number of benchmark questions: %d", len(benchmark))
    logger.info("Number of prompt cases: %d", len(prompt_cases))
    logger.info("Overall accuracy: %.4f", overall_accuracy)
    logger.info("Accuracy by prompt type: %s", accuracy_by_prompt_type)
    logger.info("Accuracy by domain: %s", accuracy_by_domain)
    logger.info("Helpful hint gain: %.4f", helpful_hint_gain)
    logger.info("Misleading hint drop: %.4f", misleading_hint_drop)
    logger.info("Saved results to %s", results_path)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info("\n%s", results_df)


if __name__ == "__main__":
    main()
