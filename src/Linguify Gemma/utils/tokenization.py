from transformers import AutoTokenizer
import torch
import pandas as pd
import os

def tokenize_data(data_path, tokenizer, max_length=128):
    """
    Tokenize a dataset using a pretrained tokenizer.

    Args:
        data_path (str): Path to the dataset CSV file.
        tokenizer: Pretrained tokenizer instance.
        max_length (int): Maximum token length.

    Returns:
        tokenized_data: Tokenized dataset in PyTorch tensor format.
    """
    df = pd.read_csv(data_path)

    # Tokenize sentences
    tokenized_data = tokenizer(
        list(df["source_sentence"]),
        text_target=list(df["target_sentence"]),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    return tokenized_data

def tokenize_all_datasets(processed_dir, tokenized_dir, model_name="gemma2"):
    """
    Tokenize all datasets in the processed directory.

    Args:
        processed_dir (str): Directory containing processed datasets.
        tokenized_dir (str): Directory to save tokenized datasets.
        model_name (str): Pretrained model name for tokenization.
    """
    os.makedirs(tokenized_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for file_name in os.listdir(processed_dir):
        if file_name.endswith(".csv"):
            input_path = os.path.join(processed_dir, file_name)
            output_path = os.path.join(tokenized_dir, f"tokenized_{file_name.replace('.csv', '.pt')}")
            tokenized_data = tokenize_data(input_path, tokenizer)
            torch.save(tokenized_data, output_path)
            print(f"Tokenized and saved: {output_path}")
