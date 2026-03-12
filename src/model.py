import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_model(features_train, target_train):
    """Train linear regression model"""
    model = LinearRegression()
    model.fit(features_train, target_train)
    return model


def evaluate_model(model, features_test, target_test):
    """Calculate RMSE and mean prediction"""
    predictions = model.predict(features_test)
    rmse = np.sqrt(mean_squared_error(target_test, predictions))
    mean_prediction = predictions.mean()
    return rmse, mean_prediction


def predict(model, features_test):
    """Return predictions"""
    predictions = model.predict(features_test)
    return predictions