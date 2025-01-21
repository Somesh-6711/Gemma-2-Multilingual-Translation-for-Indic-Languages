from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch
import os

def load_tokenized_data(tokenized_dir):
    """
    Load all tokenized datasets.

    Args:
        tokenized_dir (str): Directory containing tokenized datasets.

    Returns:
        Combined tokenized dataset.
    """
    tokenized_files = [torch.load(os.path.join(tokenized_dir, f)) for f in os.listdir(tokenized_dir) if f.endswith(".pt")]
    input_ids = torch.cat([data["input_ids"] for data in tokenized_files])
    attention_mask = torch.cat([data["attention_mask"] for data in tokenized_files])
    labels = torch.cat([data["labels"] for data in tokenized_files])
    return torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

def fine_tune_model(tokenized_dir, model_name="gemma2", output_dir="models/fine_tuned/"):
    """
    Fine-tune the translation model.

    Args:
        tokenized_dir (str): Directory with tokenized datasets.
        model_name (str): Pretrained model name.
        output_dir (str): Directory to save the fine-tuned model.
    """
    train_dataset = load_tokenized_data(tokenized_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        logging_dir="./logs",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved at {output_dir}")
