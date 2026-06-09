"""Tests for prompt generation."""

from src.prompts.prompt_generator import (
    generate_clean_prompt,
    generate_helpful_prompt,
    generate_misleading_prompt,
)


def test_generate_clean_prompt() -> None:
    """Test clean prompt generation."""
    question = "What type of object is the Sun?"

    prompt = generate_clean_prompt(
        question,
    )

    assert prompt == question


def test_generate_helpful_prompt() -> None:
    """Test helpful prompt generation."""
    prompt = generate_helpful_prompt(
        question="What type of object is the Sun?",
        helpful_hint="It produces energy by nuclear fusion.",
    )

    assert "Hint:" in prompt
    assert "nuclear fusion" in prompt


def test_generate_misleading_prompt() -> None:
    """Test misleading prompt generation."""
    prompt = generate_misleading_prompt(
        question="What type of object is the Sun?",
        misleading_hint="It is the largest planet.",
    )

    assert "Hint:" in prompt
    assert "largest planet" in prompt
