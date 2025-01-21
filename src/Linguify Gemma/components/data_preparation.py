import pandas as pd
import os
import re

def load_and_clean_data(file_path):
    """
    Load and clean the dataset.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Clean text: Remove special characters, numbers, and extra spaces
    def clean_text(text):
        text = re.sub(r"[^a-zA-Z\u0900-\u097F\s]", "", text)  # Keep English and Indic scripts
        text = re.sub(r"\s+", " ", text)  # Remove extra spaces
        return text.strip()

    df["source_sentence"] = df["source_sentence"].apply(clean_text)
    df["target_sentence"] = df["target_sentence"].apply(clean_text)

    return df

def preprocess_and_save_datasets(raw_dir, processed_dir):
    """
    Preprocess all datasets and save cleaned versions.

    Args:
        raw_dir (str): Directory containing raw datasets.
        processed_dir (str): Directory to save processed datasets.
    """
    os.makedirs(processed_dir, exist_ok=True)

    for file_name in os.listdir(raw_dir):
        if file_name.endswith(".csv"):
            print(f"Processing {file_name}...")
            file_path = os.path.join(raw_dir, file_name)
            processed_file_path = os.path.join(processed_dir, f"cleaned_{file_name}")

            # Load and clean
            df = load_and_clean_data(file_path)

            # Save cleaned dataset
            df.to_csv(processed_file_path, index=False, encoding="utf-8")
            print(f"Saved cleaned dataset to {processed_file_path}")
