import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 123


def load_data(path):
    """Load dataset"""
    data = pd.read_csv(path)
    return data


def split_features_target(data):
    """Separate features and target"""
    target = data["product"]
    features = data.drop(["id", "product"], axis=1)
    return features, target


def train_test_split_data(features, target):
    """Split data into train and test sets"""
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.25, random_state=RANDOM_STATE
    )
    return features_train, features_test, target_train, target_test


def scale_features(features_train, features_test):
    """Scale numeric features"""
    numeric = ["f0", "f1", "f2"]

    scaler = StandardScaler()
    scaler.fit(features_train[numeric])

    features_train[numeric] = scaler.transform(features_train[numeric])
    features_test[numeric] = scaler.transform(features_test[numeric])

    return features_train, features_test