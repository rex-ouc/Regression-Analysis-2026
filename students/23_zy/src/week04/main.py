import numpy as np
import time
from sklearn.metrics import mean_squared_error
from solvers import AnalyticalSolver, GradientDescentSolver
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, SGDRegressor

def generate_data(N: int, P: int, seed=42):
    np.random.seed(seed)
    X = np.random.randn(N, P)
    true_beta = np.random.randn(P + 1)
    y = true_beta[0] + X @ true_beta[1:] + np.random.randn(N) * 0.1
    return X, y, true_beta

def run_low_dim():
    print("=== 低维实验 N=10000, P=10 ===")
    X, y, _ = generate_data(10000, 10)
    start = time.time()
    ana = AnalyticalSolver().fit(X, y)
    ana_time = time.time() - start
    ana_mse = mean_squared_error(y, ana.predict(X))
    start = time.time()
    gd = GradientDescentSolver(learning_rate=0.1, epochs=5000).fit(X, y)
    gd_time = time.time() - start
    gd_mse = mean_squared_error(y, gd.predict(X))
    print(f"解析解: 时间 {ana_time:.4f}s | MSE {ana_mse:.6f}")
    print(f"梯度下降: 时间 {gd_time:.4f}s | MSE {gd_mse:.6f}")
    print("-"*50)

def run_high_dim():
    print("=== 高维实验 N=10000, P=2000 ===")
    X, y, _ = generate_data(10000, 2000)
    start = time.time()
    ana = AnalyticalSolver().fit(X, y)
    ana_time = time.time() - start
    ana_mse = mean_squared_error(y, ana.predict(X))
    start = time.time()
    gd = GradientDescentSolver(learning_rate=0.01, epochs=1000).fit(X, y)
    gd_time = time.time() - start
    gd_mse = mean_squared_error(y, gd.predict(X))
    start = time.time()
    sm_model = sm.OLS(y, sm.add_constant(X)).fit()
    sm_time = time.time() - start
    sm_mse = mean_squared_error(y, sm_model.predict(sm.add_constant(X)))
    start = time.time()
    lr = LinearRegression().fit(X, y)
    lr_time = time.time() - start
    lr_mse = mean_squared_error(y, lr.predict(X))
    start = time.time()
    sgd = SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1000).fit(X, y)
    sgd_time = time.time() - start
    sgd_mse = mean_squared_error(y, sgd.predict(X))
    print(f"解析解: 时间 {ana_time:.4f}s | MSE {ana_mse:.6f}")
    print(f"梯度下降: 时间 {gd_time:.4f}s | MSE {gd_mse:.6f}")
    print(f"statsmodels: 时间 {sm_time:.4f}s | MSE {sm_mse:.6f}")
    print(f"sklearn LR: 时间 {lr_time:.4f}s | MSE {lr_mse:.6f}")
    print(f"sklearn SGD: 时间 {sgd_time:.4f}s | MSE {sgd_mse:.6f}")

if __name__ == "__main__":
    run_low_dim()
    run_high_dim()