import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from config import MODEL_PATH
from data_loader import load_data
from preprocessing import preprocess_data
from utils import save_pickle


def train_model():
    df = load_data()
    df = preprocess_data(df)

    X = df.drop('Price', axis=1)
    y = df['Price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Save model and scaler
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)

    return model, scaler, X_test, y_test
