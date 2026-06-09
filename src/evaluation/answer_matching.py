"""Answer matching utilities."""

from fractions import Fraction

ANSWER_ALIASES: dict[str, set[str]] = {
    "yes": {"yes", "true", "correct"},
    "no": {"no", "false", "incorrect"},
    "milky way": {"milky way", "the milky way", "milky way galaxy"},
    "star": {"star", "a star"},
    "redder": {"redder", "red"},
    "moving away": {"moving away", "receding", "moves away"},
    "moving toward us": {
        "moving toward us",
        "moving towards us",
        "approaching",
        "moves toward us",
    },
    "stellar remnant": {
        "stellar remnant",
        "a stellar remnant",
        "dense remnant",
        "remnant",
    },
    "changing sun-earth-moon geometry": {
        "changing sun-earth-moon geometry",
        "relative positions of the earth moon and sun",
        "relative positions of the earth, moon, and sun",
        "relative positions of the sun earth and moon",
        "relative positions of the sun, earth, and moon",
        "sun earth moon geometry",
        "sun-earth-moon geometry",
    },
    "constant velocity": {
        "constant velocity",
        "velocity becomes constant",
        "velocity is constant",
        "moving at a constant velocity",
    },
    "gravitational potential energy": {
        "gravitational potential energy",
        "potential energy",
    },
    "supermassive black hole": {
        "supermassive black hole",
        "a supermassive black hole",
        "black hole",
    },
    "transiting exoplanet": {
        "transiting exoplanet",
        "exoplanet",
        "planet transit",
        "transit",
    },
}


def normalize_answer(
    answer: str,
) -> str:
    """Normalize answer text.

    Args:
        answer: Raw answer.

    Returns:
        Normalized answer.
    """
    normalized = answer.strip().lower()
    normalized = normalized.rstrip(".")
    normalized = " ".join(normalized.split())

    return normalized


def extract_option_letter(
    prediction: str,
) -> str | None:
    """Extract A/B/C/D answer letter from prediction.

    Args:
        prediction: Model prediction.

    Returns:
        Extracted option letter, or None if unavailable.
    """
    normalized = normalize_answer(prediction)

    if normalized in {"a", "b", "c", "d"}:
        return normalized.upper()

    prefixes = (
        "answer:",
        "final answer:",
        "the answer is",
        "option",
    )

    for prefix in prefixes:
        if normalized.startswith(prefix):
            remaining = normalized.removeprefix(prefix).strip()
            if remaining[:1] in {"a", "b", "c", "d"}:
                return remaining[:1].upper()

    if len(normalized) >= 2 and normalized[0] in {"a", "b", "c", "d"}:
        if normalized[1] in {".", ")", ":"}:
            return normalized[0].upper()

    return None


def option_letter_match(
    prediction: str,
    ground_truth: str,
) -> bool:
    """Check MCQ option-letter match.

    Args:
        prediction: Model prediction.
        ground_truth: Correct answer key.

    Returns:
        Whether prediction matches the option letter.
    """
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_ground_truth not in {"a", "b", "c", "d"}:
        return False

    predicted_letter = extract_option_letter(prediction)

    if predicted_letter is None:
        return False

    return predicted_letter == normalized_ground_truth.upper()


def exact_match(
    prediction: str,
    ground_truth: str,
) -> bool:
    """Check exact normalized match.

    Args:
        prediction: Model prediction.
        ground_truth: Expected answer.

    Returns:
        Whether prediction matches ground truth.
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def alias_match(
    prediction: str,
    ground_truth: str,
) -> bool:
    """Check answer aliases.

    Args:
        prediction: Model prediction.
        ground_truth: Expected answer.

    Returns:
        Whether prediction matches an accepted alias.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    aliases = ANSWER_ALIASES.get(
        normalized_ground_truth,
        {normalized_ground_truth},
    )

    return normalized_prediction in aliases


def numeric_match(
    prediction: str,
    ground_truth: str,
) -> bool:
    """Check simple numeric equivalence.

    Args:
        prediction: Model prediction.
        ground_truth: Expected answer.

    Returns:
        Whether numeric values are equivalent.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    try:
        prediction_value = Fraction(normalized_prediction.replace("%", ""))
        ground_truth_value = Fraction(normalized_ground_truth.replace("%", ""))
    except ValueError:
        return False

    if "%" in normalized_prediction and "%" not in normalized_ground_truth:
        prediction_value = prediction_value / 100

    if "%" in normalized_ground_truth and "%" not in normalized_prediction:
        ground_truth_value = ground_truth_value / 100

    return prediction_value == ground_truth_value


def answer_matches(
    prediction: str,
    ground_truth: str,
) -> bool:
    """Check whether prediction should be counted as correct.

    Args:
        prediction: Model prediction.
        ground_truth: Expected answer.

    Returns:
        Whether prediction matches ground truth.
    """
    if option_letter_match(
        prediction=prediction,
        ground_truth=ground_truth,
    ):
        return True

    if exact_match(
        prediction=prediction,
        ground_truth=ground_truth,
    ):
        return True

    if alias_match(
        prediction=prediction,
        ground_truth=ground_truth,
    ):
        return True

    if numeric_match(
        prediction=prediction,
        ground_truth=ground_truth,
    ):
        return True

    return False
