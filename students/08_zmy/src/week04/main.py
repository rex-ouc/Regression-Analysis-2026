"""
主实验流水线
对比低维和高维场景下不同求解器的性能
"""

import numpy as np
import time
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, SGDRegressor

from data_generator import generate_data
from solvers import AnalyticalSolver, GradientDescentSolver


def compute_mse(y_true, y_pred):
    """计算均方误差"""
    return np.mean((y_true - y_pred) ** 2)


def run_experiment(n_samples, n_features, solvers_dict):
    """
    在给定数据集上运行所有求解器，记录时间和误差

    参数:
        n_samples (int): 样本量
        n_features (int): 特征数
        solvers_dict (dict): 求解器名称到可调用对象的映射

    返回:
        results (dict): 包含各求解器的耗时和 MSE
    """
    # 生成数据
    X, y, beta_true = generate_data(n_samples, n_features, noise_std=1.0)

    # 存储结果
    results = {}

    for name, solver in solvers_dict.items():
        # 测量时间
        start = time.time()
        if name == "GradientDescentSolver (custom)":
            beta_est = solver.fit(X, y, verbose=False)
        else:
            beta_est = solver.fit(X, y)
        elapsed = time.time() - start

        # 计算预测值（注意截距项）
        if name in ["AnalyticalSolver (custom)", "GradientDescentSolver (custom)"]:
            # 自定义求解器返回的 beta 包含截距
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            y_pred = X_with_intercept @ beta_est
        elif name == "statsmodels.OLS":
            # statsmodels 需要常数项
            X_const = sm.add_constant(X)
            y_pred = solver.predict(X_const)
        else:
            # sklearn 模型（已拟合）
            y_pred = solver.predict(X)

        mse = compute_mse(y, y_pred)

        results[name] = {"time (s)": elapsed, "MSE": mse}

    return results


def main():
    # 定义实验参数
    experiments = {
        "Low-dimensional (N=10000, P=10)": (10000, 10),
        "High-dimensional (N=10000, P=2000)": (10000, 2000),
    }

    # 定义求解器
    solvers = {
        "AnalyticalSolver (custom)": AnalyticalSolver(),
        "GradientDescentSolver (custom)": GradientDescentSolver(
            learning_rate=0.01, n_epochs=500
        ),
        "statsmodels.OLS": None,  # 稍后动态创建
        "sklearn.LinearRegression": LinearRegression(),
        "sklearn.SGDRegressor": SGDRegressor(max_iter=500, tol=1e-3, random_state=42),
    }

    # 存储所有结果
    all_results = {}

    for exp_name, (n_samples, n_features) in experiments.items():
        print(f"\n{'=' * 60}")
        print(f"运行实验: {exp_name}")
        print(f"样本量={n_samples}, 特征数={n_features}")
        print(f"{'=' * 60}")

        # 对 statsmodels 单独处理，因为其接口不同
        X, y, _ = generate_data(n_samples, n_features, noise_std=1.0)

        # 临时创建 statsmodels 求解器（每次实验重新拟合）
        solvers["statsmodels.OLS"] = sm.OLS(y, sm.add_constant(X)).fit()

        # 运行所有求解器（注意：需要将数据传入，但我们的 run_experiment 会重新生成数据，
        # 所以这里我们改为单独调用每个求解器，确保它们使用同一份数据）
        results = {}

        # 自定义解析解
        start = time.time()
        beta_est = AnalyticalSolver().fit(X, y)
        elapsed = time.time() - start
        y_pred = np.column_stack([np.ones(X.shape[0]), X]) @ beta_est
        results["AnalyticalSolver (custom)"] = {
            "time (s)": elapsed,
            "MSE": compute_mse(y, y_pred),
        }

        # 自定义梯度下降
        start = time.time()
        beta_est = GradientDescentSolver(learning_rate=0.01, n_epochs=500).fit(X, y)
        elapsed = time.time() - start
        y_pred = np.column_stack([np.ones(X.shape[0]), X]) @ beta_est
        results["GradientDescentSolver (custom)"] = {
            "time (s)": elapsed,
            "MSE": compute_mse(y, y_pred),
        }

        # statsmodels
        start = time.time()
        model_sm = sm.OLS(y, sm.add_constant(X)).fit()
        elapsed = time.time() - start
        y_pred = model_sm.predict(sm.add_constant(X))
        results["statsmodels.OLS"] = {
            "time (s)": elapsed,
            "MSE": compute_mse(y, y_pred),
        }

        # sklearn LinearRegression
        start = time.time()
        model_lr = LinearRegression().fit(X, y)
        elapsed = time.time() - start
        y_pred = model_lr.predict(X)
        results["sklearn.LinearRegression"] = {
            "time (s)": elapsed,
            "MSE": compute_mse(y, y_pred),
        }

        # sklearn SGDRegressor
        start = time.time()
        model_sgd = SGDRegressor(max_iter=500, tol=1e-3, random_state=42).fit(X, y)
        elapsed = time.time() - start
        y_pred = model_sgd.predict(X)
        results["sklearn.SGDRegressor"] = {
            "time (s)": elapsed,
            "MSE": compute_mse(y, y_pred),
        }

        all_results[exp_name] = results

    # 输出结果表格
    print("\n\n" + "=" * 80)
    print("最终结果对比")
    print("=" * 80)
    for exp_name, results in all_results.items():
        print(f"\n{exp_name}:")
        df = pd.DataFrame(results).T
        print(df.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
