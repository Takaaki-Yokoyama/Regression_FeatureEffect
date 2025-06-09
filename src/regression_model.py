from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np
import pandas as pd

models = {
    'LinearRegression': LinearRegression(),
    'ElasticNet': ElasticNet(),
    'RandomForest': RandomForestRegressor(),
    'GBR': GradientBoostingRegressor(),
    'SVR': SVR()
}

def get_model(model_name: str):
    if model_name not in models:
        raise ValueError(f"未対応のモデル名です: {model_name}")
    return models[model_name]

class RegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = None
        self.target_names = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.feature_names = X.columns.tolist()
        self.target_names = y.columns.tolist()
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_from_array(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def get_coefficients(self):
        return self.model.coef_, self.model.intercept_

if __name__ == "__main__":
    # 動作確認用
    from . import data_loader
    config = data_loader.load_config("../config.json")
    df = data_loader.load_data("../data.csv")
    feature_names, target_names = data_loader.get_feature_target_names(df, config["num_targets"])
    X = df[feature_names]
    y = df[target_names]
    model = RegressionModel()
    model.fit(X, y)
    print("係数:", model.get_coefficients())
