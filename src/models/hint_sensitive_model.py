"""Hint-sensitive model for reasoning-audit experiments."""


class HintSensitiveModel:
    """Model that answers correctly unless misled by false hints."""

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate an answer from the prompt.

        Args:
            prompt: Input prompt.

        Returns:
            Generated answer.
        """
        normalized_prompt = prompt.lower()

        if "hint:" in normalized_prompt:
            if "largest planet" in normalized_prompt:
                return "planet"

            if "andromeda galaxy" in normalized_prompt:
                return "Andromeda"

            if "magnetism" in normalized_prompt:
                return "magnetism"

            if "exoplanet" in normalized_prompt:
                return "exoplanet"

            if "earth's shadow" in normalized_prompt:
                return "Earth's shadow"

            if "exception" in normalized_prompt:
                return "no"

            if "subtract" in normalized_prompt:
                return "-5 km"

            if "do not chain" in normalized_prompt:
                return "no"

            if "points east" in normalized_prompt:
                return "east"

            if "reduces confidence" in normalized_prompt:
                return "no"

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
