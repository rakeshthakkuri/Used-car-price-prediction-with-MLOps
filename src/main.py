from data_loader import load_data
from preprocessing import preprocess_data
from train import train_model
from predict import predict_single
from ensemble import custom_ensemble

# Train and evaluate
model, scaler, X_test, y_test = train_model()

# Predict using single model
sample = X_test[1]
pred = predict_single(sample)
print("Single model prediction:", pred)

# Predict using ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Need unscaled version for ensemble
df = preprocess_data(load_data())
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test_raw, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=45)

sample_raw = X_test_raw.iloc[1]
ensemble_pred = custom_ensemble(X_train, y_train, sample_raw)
print("Ensemble prediction:", ensemble_pred)
