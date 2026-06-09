"""Hugging Face sequence-to-sequence model wrapper."""

import torch
from transformers import (  # type: ignore[import-untyped]
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


class HFSeq2SeqModel:
    """Hugging Face seq2seq model for reasoning-audit experiments."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 32,
    ) -> None:
        """Initialize model.

        Args:
            model_name: Hugging Face model name.
            max_new_tokens: Maximum generated tokens.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate a model response.

        Args:
            prompt: Input prompt.

        Returns:
            Generated response.
        """
        formatted_prompt = f"Answer with only the final answer.\n\n{prompt}"

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()
