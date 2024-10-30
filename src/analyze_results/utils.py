import os
import pandas as pd
from pathlib import Path


def get_subfolders(folder_path):
    # List all subdirectories in the given folder
    subfolders = [f.name for f in Path(folder_path).iterdir() if f.is_dir()]
    return subfolders


def load_csv_in_folder(folder_path) -> pd.DataFrame:
    # List all files in the given folder
    files = os.listdir(folder_path)
    
    # Filter for CSV files
    csv_files = [file for file in files if file.endswith('.csv')]
    
    # Ensure there is exactly one CSV file
    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV files found in the folder.")
    elif len(csv_files) > 1:
        raise ValueError("Multiple CSV files found. Please ensure only one CSV file is in the folder.")
    
    # Load the CSV file into a DataFrame
    csv_file_path = os.path.join(folder_path, csv_files[0])
    df = pd.read_csv(csv_file_path)
    
    return df
