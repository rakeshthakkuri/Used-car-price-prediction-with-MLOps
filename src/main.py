from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_single
from src.ensemble import custom_ensemble
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from sklearn.model_selection import train_test_split

# Train model and evaluate
model, scaler, X_test, y_test = train_model()
evaluate_model(model, X_test, y_test)

# Predict single sample
sample = X_test[0]
print("Single model prediction:", predict_single(sample))

# Ensemble prediction
df = preprocess_data(load_data())
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test_raw, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=45)
raw_sample = X_test_raw.iloc[0].values
print("Ensemble prediction:", custom_ensemble(X_train, y_train, raw_sample))
