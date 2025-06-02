from src.data_loader import load_data
from src.preprocessing import preprocess_data
import pandas as pd

def run_pipeline_preprocessing():
    df, = load_data()
    df_processed = preprocess_data(df)
    print(f"Preprocessed data shape: {df_processed.shape}")
    # You may want to save this processed data somewhere if needed
    return df_processed

if __name__ == "__main__":
    run_pipeline_preprocessing()
