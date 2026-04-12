import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, SGDRegressor
import statsmodels.api as sm
from solvers import AnalyticalSolver, GradientDescentSolver

def generate_data(n, p):
    X = np.random.randn(n, p)
    true_beta = np.random.randn(p + 1, 1)
    X_b = np.c_[np.ones((n, 1)), X]
    y = X_b @ true_beta + np.random.normal(0, 0.1, (n, 1))
    return X, y

def run_benchmark(n, p, label):
    print(f"\n🚀 正在运行实验: {label} (N={n}, P={p})")
    X, y = generate_data(n, p)
    y_flat = y.ravel() # 部分模型需要 1D 数组
    
    results = []

    # 1. 手写解析解
    start = time.time()
    as_model = AnalyticalSolver()
    as_model.fit(X, y)
    t = time.time() - start
    mse = mean_squared_error(y, as_model.predict(X))
    results.append(["Custom Analytical", t, mse])

    # 2. 手写梯度下降 (调优 LR 以防止溢出)
    start = time.time()
    gd_model = GradientDescentSolver(learning_rate=0.1, epochs=500)
    gd_model.fit(X, y)
    t = time.time() - start
    mse = mean_squared_error(y, gd_model.predict(X))
    results.append(["Custom GD", t, mse])

    # 3. Sklearn LinearRegression (解析解)
    start = time.time()
    sk_lr = LinearRegression()
    sk_lr.fit(X, y)
    t = time.time() - start
    mse = mean_squared_error(y, sk_lr.predict(X))
    results.append(["Sklearn OLS", t, mse])

    # 4. Sklearn SGDRegressor (机器学习 GD)
    start = time.time()
    sk_sgd = SGDRegressor(max_iter=1000, tol=1e-3)
    sk_sgd.fit(X, y_flat)
    t = time.time() - start
    mse = mean_squared_error(y_flat, sk_sgd.predict(X))
    results.append(["Sklearn SGD", t, mse])

    # 5. Statsmodels OLS (传统统计)
    start = time.time()
    X_sm = sm.add_constant(X)
    sm_model = sm.OLS(y, X_sm).fit()
    t = time.time() - start
    mse = mean_squared_error(y, sm_model.predict(X_sm))
    results.append(["Statsmodels OLS", t, mse])

    df = pd.DataFrame(results, columns=["Solver", "Time (s)", "MSE"])
    print(df.to_string(index=False))

if __name__ == "__main__":
    # 实验 A: 低维场景
    run_benchmark(10000, 10, "Experiment A: Low-Dim")
    
    # 实验 B: 高维灾难
    run_benchmark(10000, 2000, "Experiment B: High-Dim")