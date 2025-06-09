import matplotlib.pyplot as plt
import numpy as np
from .regression_model import RegressionModel


def plot_feature_effect(model: RegressionModel, x_opt: np.ndarray, feature_idx: int, feature_range: np.ndarray):
    """指定した特徴量を変化させたときの目的変数の変化をプロットする"""
    xs = []
    ys = []
    for x in feature_range:
        x_input = x_opt.copy()
        x_input[feature_idx] = x
        y_pred = model.predict_from_array(x_input.reshape(1, -1))[0]
        xs.append(x)
        ys.append(y_pred)

    ys = np.array(ys)
    for i in range(ys.shape[1]):
        plt.plot(xs, ys[:, i], label=f"y{i+1}")

    plt.xlabel(f"feature {feature_idx}")
    plt.ylabel("predicted y")
    plt.legend()
    plt.show()
