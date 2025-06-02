from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from src.config import MODEL_PATH
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.utils import save_pickle
import warnings
warnings.filterwarnings('ignore')
import mlflow.sklearn

def train_model():
    df = load_data()
    
    # Only preprocess — do not save CSV here
    df = preprocess_data(df)

    # Drop target
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

    # Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MLflow logging
    mlflow.sklearn.autolog()
    model = RandomForestRegressor()
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    save_pickle({'model': model, 'scaler': scaler, 'features': X.columns.tolist()}, MODEL_PATH)

    return model, scaler, X_test_scaled, y_test
