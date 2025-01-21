import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = "Linguify Gemma"

# Define folder structure
folders = [
    "data/raw/",
    "data/processed/",
    "src/" + project_name + "/components/",
    "src/" + project_name + "/utils/",
    "src/" + project_name + "/config/",
    "research/",
]

# Define initial files to create
files = [
    "src/" + project_name + "/components/data_preparation.py",
    "src/" + project_name + "/components/model_fine_tuning.py",
    "src/" + project_name + "/utils/data_loader.py",
    "src/" + project_name + "/config/config.yaml",
    "requirements.txt",
    "README.md",
    "research/trials.ipynb",
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    logging.info(f"Created folder: {folder}")

# Create files
for file in files:
    file_path = Path(file)
    if not file_path.exists():
        with open(file_path, 'w') as f:
            f.write("")  # Create an empty file
        logging.info(f"Created file: {file}")
    else:
        logging.info(f"File already exists: {file}")
