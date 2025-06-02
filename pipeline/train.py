from src.train import run_training

def run_pipeline_training():
    print("Starting training...")
    model, scaler, X_test_scaled, y_test = run_training()
    print("Training completed.")
    return model, scaler, X_test_scaled, y_test

if __name__ == "__main__":
    run_pipeline_training()
