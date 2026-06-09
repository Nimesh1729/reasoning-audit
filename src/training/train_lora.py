"""LoRA fine-tuning utilities for FLAN-T5."""

from pathlib import Path

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.training.dataset import (
    build_hf_dataset,
    build_instruction_dataframe,
    load_training_dataframe,
)
from src.utils.logger import get_logger


def train_lora_model(
    *,
    model_name: str,
    train_csv: str | Path,
    output_dir: str | Path,
    learning_rate: float = 1e-4,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 2,
) -> None:
    """Fine-tune a seq2seq model with LoRA.

    Args:
        model_name: Hugging Face model name.
        train_csv: Path to training CSV.
        output_dir: Directory where LoRA adapter will be saved.
        learning_rate: Training learning rate.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per device.
    """
    logger = get_logger(__name__)

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q",
            "v",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    train_df = load_training_dataframe(
        train_csv,
    )

    instruction_df = build_instruction_dataframe(
        train_df,
    )

    train_dataset = build_hf_dataset(
        instruction_df,
    )

    tokenized_train_dataset = train_dataset.map(
        lambda batch: tokenizer(
            batch["input_text"],
            text_target=batch["target_text"],
            truncation=True,
            max_length=128,
        ),
        batched=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting LoRA fine-tuning.")
    trainer.train()

    logger.info("Saving LoRA adapter to %s", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
