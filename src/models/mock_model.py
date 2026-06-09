"""Mock model utilities."""


class MockModel:
    """Simple mock model for testing the evaluation pipeline."""

    def __init__(
        self,
        response: str,
    ) -> None:
        """Initialize mock model.

        Args:
            response: Fixed response returned by the model.
        """
        self.response = response

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate a fixed response.

        Args:
            prompt: Input prompt.

        Returns:
            Fixed model response.
        """
        return self.response
