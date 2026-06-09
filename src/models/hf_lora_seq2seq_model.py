"""Hugging Face LoRA seq2seq model wrapper."""

import torch
from peft import PeftModel
from transformers import (  # type: ignore[import-untyped]
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


class HFLoraSeq2SeqModel:
    """Seq2seq model with a loaded LoRA adapter."""

    def __init__(
        self,
        model_name: str,
        adapter_path: str,
        max_new_tokens: int = 32,
    ) -> None:
        """Initialize base model with LoRA adapter.

        Args:
            model_name: Base Hugging Face model name.
            adapter_path: Path to saved LoRA adapter.
            max_new_tokens: Maximum generated tokens.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
        )

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
        )

        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate a response.

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
