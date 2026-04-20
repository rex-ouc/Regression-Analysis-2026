import numpy as np
from data_generator import generate_design_matrix

# ===================== 全局参数设置 =====================
np.random.seed(123)  # 固定随机种子，结果可复现
n_samples = 1000     # 样本量
n_simulations = 1000 # 蒙特卡洛模拟次数
beta_true = np.array([5.0, 3.0])  # 真实参数
sigma = 2.0          # 噪音标准差

# ===================== 实验A：独立特征 (rho=0) =====================
X_A = generate_design_matrix(n_samples, rho=0.0)
betas_hat_A = []

for _ in range(n_simulations):
    # 生成随机噪音
    epsilon = np.random.normal(0, sigma, n_samples)
    # 生成因变量 y
    y = X_A @ beta_true + epsilon
    # 最小二乘估计 β = (X^T X)^{-1} X^T y
    beta_hat = np.linalg.inv(X_A.T @ X_A) @ X_A.T @ y
    betas_hat_A.append(beta_hat)

betas_hat_A = np.array(betas_hat_A)

# ===================== 实验B：高度共线性 (rho=0.99) =====================
X_B = generate_design_matrix(n_samples, rho=0.99)
betas_hat_B = []

for _ in range(n_simulations):
    epsilon = np.random.normal(0, sigma, n_samples)
    y = X_B @ beta_true + epsilon
    beta_hat = np.linalg.inv(X_B.T @ X_B) @ X_B.T @ y
    betas_hat_B.append(beta_hat)

betas_hat_B = np.array(betas_hat_B)

# ===================== 任务3：理论协方差矩阵 vs 经验协方差矩阵（实验B） =====================
# 1. 经验协方差矩阵（1000次模拟结果计算）
empirical_cov = np.cov(betas_hat_B, rowvar=False)

# 2. 理论协方差矩阵：σ² * (X^T X)^{-1}
theoretical_cov = sigma ** 2 * np.linalg.inv(X_B.T @ X_B)

# 打印结果
print("=" * 60)
print("实验B（rho=0.99）：经验协方差矩阵")
print(np.round(empirical_cov, 4))
print("=" * 60)
print("实验B（rho=0.99）：理论协方差矩阵")
print(np.round(theoretical_cov, 4))
print("=" * 60)
