import numpy as np
from scipy.optimize import minimize
from . import data_loader
from . import regression_model


def multi_objective_func(x, model, objectives):
    # x: 1次元配列, model: RegressionModel, objectives: ["maximize", "minimize", ...]
    x = np.array(x).reshape(1, -1)
    y_pred = model.predict_from_array(x)[0]  # shape: (num_targets,)
    # maximize→-y, minimize→+y でスカラー化
    result = 0
    for i, obj in enumerate(objectives):
        if obj == "maximize":
            result -= y_pred[i]
        else:
            result += y_pred[i]
    return result


def inverse_analysis(model, config):
    bounds = config["x_bounds"]
    objectives = config["objectives"]
    x0 = np.array([(b[0]+b[1])/2 for b in bounds])
    res = minimize(
        multi_objective_func,
        x0,
        args=(model, objectives),
        bounds=bounds,
        method='L-BFGS-B',
    )
    return res.x, res.fun



if __name__ == "__main__":
    config = data_loader.load_config("../config.json")
    df = data_loader.load_data("../data.csv")
    feature_names, target_names = data_loader.get_feature_target_names(df, config["num_targets"])
    X = df[feature_names]
    y = df[target_names]
    model = regression_model.RegressionModel()
    model.fit(X, y)
    x_opt, fval = inverse_analysis(model, config)
    print("最適X:", x_opt)
    print("目的関数値:", fval)
