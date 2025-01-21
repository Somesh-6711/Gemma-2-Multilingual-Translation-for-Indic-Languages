import pandas as pd
import os
import re

def clean_text(text):
    """
    Clean text by removing special characters, numbers, and extra spaces.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r"[^a-zA-Z\u0900-\u097F\s]", "", text)  # Keep English and Indic scripts
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    return text.strip()

def preprocess_dataset(input_file, output_file):
    """
    Load and preprocess a single dataset.

    Args:
        input_file (str): Path to the raw dataset CSV file.
        output_file (str): Path to save the preprocessed dataset CSV file.
    """
    df = pd.read_csv(input_file)

    # Remove duplicates and rows with missing values
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Clean text
    df["source_sentence"] = df["source_sentence"].apply(clean_text)
    df["target_sentence"] = df["target_sentence"].apply(clean_text)

    # Save cleaned dataset
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Processed and saved: {output_file}")

def preprocess_all_datasets(raw_dir, processed_dir):
    """
    Preprocess all datasets in the raw directory.

    Args:
        raw_dir (str): Directory containing raw datasets.
        processed_dir (str): Directory to save processed datasets.
    """
    os.makedirs(processed_dir, exist_ok=True)

    for file_name in os.listdir(raw_dir):
        if file_name.endswith(".csv"):
            input_file = os.path.join(raw_dir, file_name)
            output_file = os.path.join(processed_dir, f"cleaned_{file_name}")
            preprocess_dataset(input_file, output_file)
