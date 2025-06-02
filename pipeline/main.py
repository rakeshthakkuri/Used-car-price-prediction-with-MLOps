import os
import sys

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from train import train_model
from evaluate import evaluate_model
from predict import predict_single
from ensemble import custom_ensemble
from data_loader import load_data
from preprocessing import preprocess_data
from sklearn.model_selection import train_test_split


def main():
    # 1. Train the model and get scaler, test data
    model, scaler, X_test_scaled, y_test = train_model()

    # 2. Evaluate the model
    evaluate_model(model, X_test_scaled, y_test)

    # 3. Predict single sample from test data
    sample_scaled = X_test_scaled[0]
    single_pred = predict_single(sample_scaled)
    print(f"Single model prediction: {single_pred}")

    # 4. Ensemble prediction using raw (unscaled) data and train/test split
    df = preprocess_data(load_data())
    X = df.drop('Price', axis=1)
    y = df['Price']

    X_train_raw, X_test_raw, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=45)
    raw_sample = X_test_raw.iloc[0].values  # raw sample, unscaled features

    ensemble_pred = custom_ensemble(X_train_raw, y_train, raw_sample)
    print(f"Ensemble prediction: {ensemble_pred}")

if __name__ == "__main__":
    main()
