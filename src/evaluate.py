import json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
from config import METRICS_PATH
import os

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        'R2 Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': sqrt(mean_squared_error(y_test, y_pred))
    }

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)

    for key, value in metrics.items():
        print(f'{key}: {value:.4f}')
