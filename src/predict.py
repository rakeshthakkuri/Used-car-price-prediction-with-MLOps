import pickle
import numpy as np
from config import MODEL_PATH

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler']

def predict_single(sample):
    """
    sample: a single data point as a NumPy array or list, already preprocessed (same order as training features)
    """
    model, scaler = load_model()
    sample_scaled = scaler.transform([sample])
    prediction = model.predict(sample_scaled)
    return prediction[0]
