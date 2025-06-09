import numpy as np
from pathlib import Path
# 直接実行した際にも読み込めるよう、絶対インポートを使用
from src import data_loader, regression_model, inverse_analysis, visualization


def run(model_name: str = "LinearRegression", feature_idx: int = 0, num_points: int = 50):
    root = Path(__file__).resolve().parent.parent
    config = data_loader.load_config(root / "config.json")
    df = data_loader.load_data(root / "data.csv")
    feature_names, target_names = data_loader.get_feature_target_names(df, config["num_targets"])
    X = df[feature_names]
    y = df[target_names]

    model = regression_model.RegressionModel()
    model.model = regression_model.get_model(model_name)
    model.fit(X, y)

    x_opt, _ = inverse_analysis.inverse_analysis(model, config)

    feature_range = np.linspace(config["x_bounds"][feature_idx][0], config["x_bounds"][feature_idx][1], num_points)
    visualization.plot_feature_effect(model, x_opt, feature_idx, feature_range)


if __name__ == "__main__":
    run()
