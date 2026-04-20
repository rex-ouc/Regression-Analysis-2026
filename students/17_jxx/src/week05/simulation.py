import numpy as np
from data_generator import generate_design_matrix

# ===================== 实验参数 =====================
np.random.seed(42)
n_samples = 1000
n_simulation = 1000
beta_true = np.array([5.0, 3.0])
sigma = 2.0

# ===================== 实验 A：正交特征 ρ=0 =====================
X_A = generate_design_matrix(n_samples, rho=0.0)
betas_A = []

for _ in range(n_simulation):
    eps = np.random.normal(0, sigma, n_samples)
    y = X_A @ beta_true + eps
    beta_hat = np.linalg.inv(X_A.T @ X_A) @ X_A.T @ y
    betas_A.append(beta_hat)

betas_A = np.array(betas_A)

# ===================== 实验 B：高度共线性 ρ=0.99 =====================
X_B = generate_design_matrix(n_samples, rho=0.99)
betas_B = []

for _ in range(n_simulation):
    eps = np.random.normal(0, sigma, n_samples)
    y = X_B @ beta_true + eps
    beta_hat = np.linalg.inv(X_B.T @ X_B) @ X_B.T @ y
    betas_B.append(beta_hat)

betas_B = np.array(betas_B)

# ===================== 理论 vs 经验协方差矩阵 =====================
empirical_cov = np.cov(betas_B, rowvar=False)
theoretical_cov = sigma ** 2 * np.linalg.inv(X_B.T @ X_B)

# ===================== 打印输出 =====================
print("=" * 60)
print("Experiment B (ρ=0.99) Empirical Covariance Matrix")
print(empirical_cov)
print("=" * 60)
print("Experiment B (ρ=0.99) Theoretical Covariance Matrix")
print(theoretical_cov)
print("=" * 60)