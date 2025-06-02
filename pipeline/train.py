import os
import sys

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from train import train_model

def run_pipeline_training():
    print("Starting training...")
    model, scaler, X_test_scaled, y_test = train_model()
    print("Training completed.")
    return model, scaler, X_test_scaled, y_test

if __name__ == "__main__":
    run_pipeline_training()
