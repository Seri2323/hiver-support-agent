import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

MODEL_NAME = "distilbert/distilgpt2"
MAX_LENGTH = 256

print("Loading datasets...")

train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")

train_df = train_df[["customer_message", "response"]]
val_df = val_df[["customer_message", "response"]]

def format_example(row):
    return f"Customer: {row['customer_message']}\nAgent: {row['response']}"

train_df["text"] = train_df.apply(format_example, axis=1)
val_df["text"] = val_df.apply(format_example, axis=1)

train_dataset = Dataset.from_pandas(train_df[["text"]])
val_dataset = Dataset.from_pandas(val_df[["text"]])

print(f"Train examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}")

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

train_dataset = train_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.config.pad_token_id = tokenizer.pad_token_id

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Training device: {device}")

training_args = TrainingArguments(
    output_dir="reports/model",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    report_to="none",
    use_cpu=(device == "cpu"),
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("Starting training...")

trainer.train()

print("Training complete!")

trainer.save_model("reports/model/final")
tokenizer.save_pretrained("reports/model/final")

print("Model saved to reports/model/final")