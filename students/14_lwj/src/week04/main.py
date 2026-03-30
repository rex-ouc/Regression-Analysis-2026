import numpy as np
import time
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, SGDRegressor
from solvers import AnalyticalSolver, GradientDescentSolver

def generate_data(N: int, P: int, seed: int = 42):
    np.random.seed(seed)
    X = np.random.randn(N, P)
    beta_true = np.random.randn(P + 1)
    y = beta_true[0] + X @ beta_true[1:] + np.random.randn(N) * 0.5
    return X, y, beta_true

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

print("=" * 50)
print("实验1：低维 vs 高维 手写求解器对比")
print("=" * 50)

X_low, y_low, _ = generate_data(N=10000, P=10)
X_high, y_high, _ = generate_data(N=10000, P=2000)

# 解析解 低维
solver = AnalyticalSolver()
start = time.time()
solver.fit(X_low, y_low)
t_ana_low = time.time() - start
mse_ana_low = compute_mse(y_low, solver.predict(X_low))

# 解析解 高维
solver = AnalyticalSolver()
start = time.time()
solver.fit(X_high, y_high)
t_ana_high = time.time() - start
mse_ana_high = compute_mse(y_high, solver.predict(X_high))

# 梯度下降 低维
solver = GradientDescentSolver(learning_rate=1e-3, epochs=1000)
start = time.time()
solver.fit(X_low, y_low)
t_gd_low = time.time() - start
mse_gd_low = compute_mse(y_low, solver.predict(X_low))

# 梯度下降 高维
solver = GradientDescentSolver(learning_rate=1e-5, epochs=1000)
start = time.time()
solver.fit(X_high, y_high)
t_gd_high = time.time() - start
mse_gd_high = compute_mse(y_high, solver.predict(X_high))

print("\n低维场景 (N=10000, P=10)")
print(f"解析解：时间={t_ana_low:.4f}s | MSE={mse_ana_low:.4f}")
print(f"梯度下降：时间={t_gd_low:.4f}s | MSE={mse_gd_low:.4f}")

print("\n高维场景 (N=10000, P=2000)")
print(f"解析解：时间={t_ana_high:.4f}s | MSE={mse_ana_high:.4f}")
print(f"梯度下降：时间={t_gd_high:.4f}s | MSE={mse_gd_high:.4f}")

print("\n" + "=" * 50)
print("实验2：高维场景 工业库速度对比")
print("=" * 50)

# statsmodels
start = time.time()
X_sm = sm.add_constant(X_high)
sm.OLS(y_high, X_sm).fit()
t_sm = time.time() - start

# sklearn LinearRegression
start = time.time()
LinearRegression().fit(X_high, y_high)
t_sk = time.time() - start

# sklearn SGDRegressor
start = time.time()
SGDRegressor(eta0=1e-5, max_iter=1000).fit(X_high, y_high)
t_sgd = time.time() - start

print(f"statsmodels OLS: {t_sm:.4f}s")
print(f"sklearn LinearRegression: {t_sk:.4f}s")
print(f"sklearn SGDRegressor: {t_sgd:.4f}s")

print("\n✅ 第四周作业运行完成！")