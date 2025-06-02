from src.predict import predict_single

def run_pipeline_prediction(sample):
    prediction = predict_single(sample)
    print(f"Prediction: {prediction}")
    return prediction

if __name__ == "__main__":
    # Example usage with dummy sample, replace with actual feature array
    sample = [10, 1200, 100, 5, 0, 1, 0]  # Replace with proper sample features
    run_pipeline_prediction(sample)
