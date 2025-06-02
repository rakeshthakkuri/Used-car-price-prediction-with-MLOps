from config import MODEL_PATH
from utils import load_pickle
import numpy as np

def predict_single(sample):
    """
    Predict price for a single sample (list or 1D NumPy array).
    """
    data = load_pickle(MODEL_PATH)
    model = data['model']
    scaler = data['scaler']

    sample_scaled = scaler.transform([sample])
    prediction = model.predict(sample_scaled)
    return prediction[0]
