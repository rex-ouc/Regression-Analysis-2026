import numpy as np
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, SGDRegressor
import statsmodels.api as sm
from solvers import AnalyticalSolver, GradientDescentSolver


def run_test(N, P):
    # 数据生成
    X = np.random.randn(N, P)
    true_beta = np.random.randn(P + 1)
    y = np.c_[np.ones((N, 1)), X].dot(true_beta) + np.random.randn(N) * 0.5

    stats = {}

    # 手写解析法
    t0 = time.time()
    m1 = AnalyticalSolver()
    m1.fit(X, y)
    stats["Custom_Analytical"] = time.time() - t0

    # 手写梯度下降
    t0 = time.time()
    m2 = GradientDescentSolver(epochs=500)
    m2.fit(X, y)
    stats["Custom_GD"] = time.time() - t0

    # Statsmodels
    t0 = time.time()
    X_sm = sm.add_constant(X)
    sm.OLS(y, X_sm).fit()
    stats["Statsmodels_OLS"] = time.time() - t0

    # Sklearn Linear
    t0 = time.time()
    LinearRegression().fit(X, y)
    stats["Sklearn_Linear"] = time.time() - t0

    # Sklearn SGD
    t0 = time.time()
    SGDRegressor(max_iter=500).fit(X, y)
    stats["Sklearn_SGD"] = time.time() - t0

    return stats


# 执行实验
print("--- 实验 A (P=10) ---")
res_a = run_test(10000, 10)
for k, v in res_a.items():
    print(f"{k}: {v:.4f}s")

print("\n--- 实验 B (P=2000) ---")
res_b = run_test(10000, 2000)
for k, v in res_b.items():
    print(f"{k}: {v:.4f}s")
