from transformers import AutoTokenizer
import pandas as pd
import torch
import os

def tokenize_data(data_path, tokenizer, max_length):
    """
    Tokenize a dataset using the specified tokenizer.
    """
    df = pd.read_csv(data_path)

    tokenized_data = tokenizer(
        list(df["source_sentence"]),
        text_target=list(df["target_sentence"]),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    return tokenized_data

def tokenize_all_datasets(processed_dir, tokenized_dir, model_name, max_length):
    """
    Tokenize all preprocessed datasets.
    """
    os.makedirs(tokenized_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for file_name in os.listdir(processed_dir):
        if file_name.endswith(".csv"):
            input_path = os.path.join(processed_dir, file_name)
            output_path = os.path.join(tokenized_dir, f"tokenized_{file_name.replace('.csv', '.pt')}")
            tokenized_data = tokenize_data(input_path, tokenizer, max_length)
            torch.save(tokenized_data, output_path)
            print(f"Tokenized and saved: {output_path}")
