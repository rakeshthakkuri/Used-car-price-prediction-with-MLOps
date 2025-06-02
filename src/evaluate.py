from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
from src.config import METRICS_PATH
from src.utils import save_json, print_metrics

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "R2 Score": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": sqrt(mean_squared_error(y_test, y_pred))
    }

    save_json(metrics, METRICS_PATH)
    print_metrics(metrics)
    return metrics
