import time
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, SGDRegressor
from solvers import AnalyticalSolver, GradientDescentSolver


def generate_data(n_samples, n_features, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    true_beta = np.random.randn(n_features + 1)
    y = X @ true_beta[1:] + true_beta[0] + 0.1 * np.random.randn(n_samples)
    return X, y


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def main():
    print("=" * 60)
    print("低维场景 N=10000, P=10")
    X_low, y_low = generate_data(10000, 10)

    start = time.time()
    ana = AnalyticalSolver().fit(X_low, y_low)
    t_ana_low = time.time() - start

    start = time.time()
    gd = GradientDescentSolver().fit(X_low, y_low)
    t_gd_low = time.time() - start

    print(f"解析解耗时: {t_ana_low:.4f}s")
    print(f"梯度下降耗时: {t_gd_low:.4f}s\n")

    print("=" * 60)
    print("高维场景 N=10000, P=2000")
    X_high, y_high = generate_data(10000, 2000)

    start = time.time()
    ana.fit(X_high, y_high)
    t_ana_high = time.time() - start

    start = time.time()
    gd.fit(X_high, y_high)
    t_gd_high = time.time() - start

    start = time.time()
    sm.OLS(y_high, np.hstack([np.ones((len(X_high), 1)), X_high])).fit()
    t_sm = time.time() - start

    start = time.time()
    LinearRegression().fit(X_high, y_high)
    t_sk = time.time() - start

    start = time.time()
    SGDRegressor(max_iter=50000, tol=1e-6).fit(X_high, y_high)
    t_sgd = time.time() - start

    print(f"自定义解析解: {t_ana_high:.4f}s")
    print(f"自定义梯度下降: {t_gd_high:.4f}s")
    print(f"statsmodels.OLS: {t_sm:.4f}s")
    print(f"sklearn LR: {t_sk:.4f}s")
    print(f"sklearn SGD: {t_sgd:.4f}s")


if __name__ == "__main__":
    main()
