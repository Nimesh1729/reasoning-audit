"""Rule-based model for deterministic benchmark testing."""


class RuleBasedModel:
    """Simple keyword-based model for reasoning-audit smoke tests."""

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate a deterministic answer.

        Args:
            prompt: Input prompt.

        Returns:
            Rule-based answer.
        """
        normalized_prompt = prompt.lower()

        if "what type of object is the sun" in normalized_prompt:
            return "star"

        if "which galaxy contains the solar system" in normalized_prompt:
            return "Milky Way"

        if "force keeps planets" in normalized_prompt:
            return "gravity"

        if "what is a quasar" in normalized_prompt:
            return "active galactic nucleus"

        if "phases of the moon" in normalized_prompt:
            return "changing Sun-Earth-Moon geometry"

        if (
            "sun luminous" in normalized_prompt
            or "is the sun luminous" in normalized_prompt
        ):
            return "yes"

        if "10 km" in normalized_prompt and "15 km" in normalized_prompt:
            return "25 km"

        if "a is greater than b" in normalized_prompt:
            return "yes"

        if "180 degrees" in normalized_prompt:
            return "south"

        if "independent observations agree" in normalized_prompt:
            return "yes"

        return "unknown"
