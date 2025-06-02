from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from config import MODEL_PATH
from data_loader import load_data
from preprocessing import preprocess_data
from utils import save_pickle
import warnings
warnings.filterwarnings('ignore')

def train_model():
    df = load_data()
    df = preprocess_data(df)

    X = df.drop('Price', axis=1)
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor()
    model.fit(X_train_scaled, y_train)

    save_pickle({'model': model, 'scaler': scaler}, MODEL_PATH)

    return model, scaler, X_test_scaled, y_test
