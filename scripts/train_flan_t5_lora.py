"""Train FLAN-T5 with LoRA."""

from src.training.train_lora import train_lora_model
from src.utils.cli import parse_args
from src.utils.config_loader import load_config


def main() -> None:
    """Run LoRA training."""
    args = parse_args()
    config = load_config(args.config)

    train_lora_model(
        model_name=config["model"]["name"],
        train_csv=config["data"]["train_csv"],
        output_dir=config["training"]["output_dir"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    )


if __name__ == "__main__":
    main()
