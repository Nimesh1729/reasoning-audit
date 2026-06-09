"""Semi-robust model for reasoning-audit experiments."""


class SemiRobustModel:
    """Model that resists some misleading hints but not all."""

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate an answer.

        Args:
            prompt: Input prompt.

        Returns:
            Predicted answer.
        """
        normalized_prompt = prompt.lower()

        # Resist astronomy misinformation
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

        # Vulnerable to logic misinformation
        if "exception to the rule" in normalized_prompt:
            return "no"

        if "subtract the second distance" in normalized_prompt:
            return "-5 km"

        if "do not chain" in normalized_prompt:
            return "no"

        if "points east" in normalized_prompt:
            return "east"

        if "reduces confidence" in normalized_prompt:
            return "no"

        # Clean logic answers
        if "sun luminous" in normalized_prompt:
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
