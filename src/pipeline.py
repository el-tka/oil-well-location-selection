from src.preprocessing import *
from src.model import *
from src.bootstrap import *

# загрузка данных
data = load_data("data/geo_data_0.csv")

# подготовка данных
features, target = split_features_target(data)

X_train, X_test, y_train, y_test = train_test_split_data(features, target)

X_train, X_test = scale_features(X_train, X_test)

# обучение модели
model = train_model(X_train, y_train)

rmse, mean_pred = evaluate_model(model, X_test, y_test)

# предсказания
predictions = predict(model, X_test)

# bootstrap анализ
mean_profit, ci, risk, values = bootstrap_profit(y_test, predictions)