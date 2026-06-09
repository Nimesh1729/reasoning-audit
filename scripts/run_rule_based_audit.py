"""Run a rule-based reasoning-audit experiment."""

import pandas as pd

from src.evaluation.benchmark_dataset import build_prompt_cases
from src.evaluation.benchmark_loader import load_benchmark
from src.evaluation.evaluator import evaluate_prompt_cases
from src.evaluation.metrics import (
    compute_accuracy,
    compute_accuracy_by_domain,
    compute_accuracy_by_prompt_type,
)
from src.models.rule_based_model import RuleBasedModel
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


def main() -> None:
    """Run rule-based audit pipeline."""
    logger = get_logger(__name__)
    args = parse_args()

    config = load_config(args.config)
    benchmark_csv = config["data"]["benchmark_csv"]

    benchmark = load_benchmark(benchmark_csv)
    prompt_cases = build_prompt_cases(benchmark)

    model = RuleBasedModel()

    results = evaluate_prompt_cases(
        prompt_cases=prompt_cases,
        model=model,
    )

    overall_accuracy = compute_accuracy(results)
    accuracy_by_prompt_type = compute_accuracy_by_prompt_type(results)
    accuracy_by_domain = compute_accuracy_by_domain(results)

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

    logger.info("Number of benchmark questions: %d", len(benchmark))
    logger.info("Number of prompt cases: %d", len(prompt_cases))
    logger.info("Overall accuracy: %.4f", overall_accuracy)
    logger.info("Accuracy by prompt type: %s", accuracy_by_prompt_type)
    logger.info("Accuracy by domain: %s", accuracy_by_domain)
    logger.info("\n%s", results_df)


if __name__ == "__main__":
    main()
