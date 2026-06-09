"""Hugging Face causal language model wrapper."""

from transformers import (  # type: ignore[import-untyped]
    AutoModelForCausalLM,
    AutoTokenizer,
)


class HFCausalLMModel:
    """Hugging Face causal LM for reasoning-audit experiments."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 32,
    ) -> None:
        """Initialize causal language model.

        Args:
            model_name: Hugging Face model name.
            max_new_tokens: Maximum number of generated tokens.
        """
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate a response.

        Args:
            prompt: Input prompt.

        Returns:
            Generated answer.
        """
        messages = [
            {
                "role": "system",
                "content": "Answer with only the final answer.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]

        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
