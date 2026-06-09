from src.training.dataset import (
    build_instruction_dataframe,
    load_training_dataframe,
)

df = load_training_dataframe("data/benchmark/train.csv")

instruction_df = build_instruction_dataframe(df)

print(instruction_df.head())
