from src.components.data_preparation import preprocess_and_save_datasets

if __name__ == "__main__":
    raw_dir = "data/raw/"
    processed_dir = "data/processed/"

    preprocess_and_save_datasets(raw_dir, processed_dir)
