"""Command-line utilities."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse common command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration file.",
    )

    return parser.parse_args()
