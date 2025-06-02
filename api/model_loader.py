import pickle
import pandas as pd
from src.config import MODEL_PATH

with open(MODEL_PATH, 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
scaler = artifacts['scaler']
feature_order = artifacts['features']

def preprocess_input(input_dict: dict):
    df = pd.DataFrame([input_dict])

    # Add missing columns with 0
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0

    # Keep columns in the exact order
    df = df[feature_order]

    X_scaled = scaler.transform(df)
    return X_scaled
