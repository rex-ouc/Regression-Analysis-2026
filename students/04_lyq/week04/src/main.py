import numpy as np
import time

from solvers import AnalyticalSolver, GradientDescentSolver

from sklearn.linear_model import LinearRegression, SGDRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


def generate_data(N, P):
    np.random.seed(42)
    X = np.random.randn(N, P)
    true_beta = np.random.randn(P)
    y = X @ true_beta + np.random.randn(N) * 0.1
    return X, y


def evaluate(model, X, y, name):
    start = time.time()

    if name == "statsmodels":
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        y_pred = model.predict(X_sm)
    else:
        model.fit(X, y)
        y_pred = model.predict(X)

    end = time.time()

    mse = mean_squared_error(y, y_pred)

    return end - start, mse


def run_experiment(N, P):
    print(f"\n===== N={N}, P={P} =====")

    X, y = generate_data(N, P)

    results = {}

    # 自己实现
    results["AnalyticalSolver"] = evaluate(
        AnalyticalSolver(), X, y, "analytical"
    )
    results["GradientDescentSolver"] = evaluate(
        GradientDescentSolver(lr=0.01, epochs=500), X, y, "gd"
    )

    # 工业 API
    results["LinearRegression"] = evaluate(
        LinearRegression(), X, y, "sklearn"
    )

    results["SGDRegressor"] = evaluate(
        SGDRegressor(max_iter=1000, tol=1e-3), X, y, "sgd"
    )

    results["Statsmodels OLS"] = evaluate(
        None, X, y, "statsmodels"
    )

    # 输出结果
    for k, v in results.items():
        print(f"{k:25s} | Time: {v[0]:.4f}s | MSE: {v[1]:.6f}")

    return results


if __name__ == "__main__":
    # 实验 A：低维
    run_experiment(10_000, 10)

    # 实验 B：高维
    run_experiment(10_000, 2000)