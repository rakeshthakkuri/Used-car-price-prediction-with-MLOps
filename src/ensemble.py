import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression

def custom_ensemble(x_train, y_train, x_test_sample):
    """
    x_test_sample: a single test sample (unscaled)
    """
    models = [
        RandomForestRegressor(),
        Ridge(),
        LinearRegression()
    ]

    predictions = []
    for model in models:
        model.fit(x_train, y_train)
        predictions.append(model.predict([x_test_sample]))

    ensemble_prediction = np.mean(predictions)
    return ensemble_prediction
