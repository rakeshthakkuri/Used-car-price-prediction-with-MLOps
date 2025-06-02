import os
import sys

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from data_loader import load_data
from preprocessing import preprocess_data
import pandas as pd

def run_pipeline_preprocessing():
    df = load_data()
    df_processed = preprocess_data(df)

    print(f"Preprocessed data shape: {df_processed.shape}")
    # You may want to save this processed data somewhere if needed
    df_processed.to_csv("data/processed/processed_data.csv", index=False)
    return df_processed

if __name__ == "__main__":
    run_pipeline_preprocessing()
