import time
import numpy as np
from src.solvers import AnalyticalSolver, GradientDescentSolver

def generate_data(n_samples, n_features, noise_std=0.1):
    """生成线性回归数据：y = Xβ_true + ε"""
    X = np.random.randn(n_samples, n_features)  # 特征矩阵（均值0，方差1）
    beta_true = np.random.randn(n_features + 1) # 真实系数（含截距）
    X_with_intercept = np.hstack([np.ones((n_samples, 1)), X])
    y = X_with_intercept @ beta_true + np.random.randn(n_samples) * noise_std
    return X, y, beta_true

def evaluate_solver(solver, X_train, y_train, X_test, y_test):
    """评估求解器：记录耗时和 MSE 精度"""
    # 训练耗时
    start_time = time.time()
    solver.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 预测与精度（MSE）
    y_pred = solver.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    
    return train_time, mse

# ========== 实验A：低维场景 (N=10^4, P=10) ==========
n_samples = 10_000
n_features_low = 10
X_low, y_low, beta_true_low = generate_data(n_samples, n_features_low)

# 划分训练集/测试集（80%训练，20%测试）
split_idx = int(0.8 * n_samples)
X_train_low, X_test_low = X_low[:split_idx], X_low[split_idx:]
y_train_low, y_test_low = y_low[:split_idx], y_low[split_idx:]

# 初始化求解器
analytical_low = AnalyticalSolver()
gd_low = GradientDescentSolver(learning_rate=0.01, epochs=1000)

# 评估
analytical_time_low, analytical_mse_low = evaluate_solver(analytical_low, X_train_low, y_train_low, X_test_low, y_test_low)
gd_time_low, gd_mse_low = evaluate_solver(gd_low, X_train_low, y_train_low, X_test_low, y_test_low)

# ========== 实验B：高维灾难 (N=10^4, P=2000) ==========
n_features_high = 2000
X_high, y_high, beta_true_high = generate_data(n_samples, n_features_high)

# 划分训练集/测试集
X_train_high, X_test_high = X_high[:split_idx], X_high[split_idx:]
y_train_high, y_test_high = y_high[:split_idx], y_high[split_idx:]

# 初始化求解器
analytical_high = AnalyticalSolver()
gd_high = GradientDescentSolver(learning_rate=0.01, epochs=1000)

# 评估（注意：高维下解析解可能很慢或内存溢出）
try:
    analytical_time_high, analytical_mse_high = evaluate_solver(analytical_high, X_train_high, y_train_high, X_test_high, y_test_high)
except Exception as e:
    analytical_time_high = "OOM (Out of Memory)"
    analytical_mse_high = None
gd_time_high, gd_mse_high = evaluate_solver(gd_high, X_train_high, y_train_high, X_test_high, y_test_high)

# 打印结果
print("=== 低维场景 (P=10) ===")
print(f"AnalyticalSolver: 时间={analytical_time_low:.4f}s, MSE={analytical_mse_low:.4f}")
print(f"GradientDescentSolver: 时间={gd_time_low:.4f}s, MSE={gd_mse_low:.4f}")

print("\n=== 高维场景 (P=2000) ===")
print(f"AnalyticalSolver: 时间={analytical_time_high}, MSE={analytical_mse_high}")
print(f"GradientDescentSolver: 时间={gd_time_high:.4f}s, MSE={gd_mse_high:.4f}")