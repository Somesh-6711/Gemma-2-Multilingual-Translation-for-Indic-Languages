import pandas as pd
import os
import re

def clean_text(text):
    """
    Clean text by removing special characters and extra spaces.
    """
    text = re.sub(r"[^a-zA-Z\u0900-\u097F\s]", "", text)  # Keep English and Indic scripts
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    return text.strip()

def preprocess_dataset(input_file, output_file):
    """
    Preprocess a single dataset by cleaning the text and removing duplicates.
    """
    df = pd.read_csv(input_file)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    df["source_sentence"] = df["source_sentence"].apply(clean_text)
    df["target_sentence"] = df["target_sentence"].apply(clean_text)

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Preprocessed and saved: {output_file}")

def preprocess_all_datasets(raw_dir, processed_dir):
    """
    Preprocess all datasets in the raw directory.
    """
    os.makedirs(processed_dir, exist_ok=True)

    for file_name in os.listdir(raw_dir):
        if file_name.endswith(".csv"):
            input_path = os.path.join(raw_dir, file_name)
            output_path = os.path.join(processed_dir, f"cleaned_{file_name}")
            preprocess_dataset(input_path, output_path)
