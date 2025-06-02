import os
import sys

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from predict import predict_single

def run_pipeline_prediction(sample):
    prediction = predict_single(sample)
    print(f"Prediction: {prediction}")
    return prediction

if __name__ == "__main__":
    # Example usage with dummy sample, replace with actual feature array
    sample = [10, 1200, 100, 5, 0, 1, 0]  # Replace with proper sample features
    run_pipeline_prediction(sample)
