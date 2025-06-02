import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression

def custom_ensemble(X_train, y_train, X_sample):
    """
    X_sample: 1D unscaled feature array (not preprocessed)
    Returns mean prediction from multiple models
    """
    models = [
        RandomForestRegressor(),
        Ridge(),
        LinearRegression()
    ]

    predictions = []
    for model in models:
        model.fit(X_train, y_train)
        predictions.append(model.predict([X_sample]))

    return np.mean(predictions)
