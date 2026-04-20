# simulation.py
import numpy as np
from data_generator import generate_X

def ols_beta(X, y):
    """OLS 公式求解 beta_hat"""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def run_simulation(rho=0.0, n_samples=100, n_sim=1000, beta_true=np.array([5.0, 3.0]), sigma=2.0, seed=None):
    X = generate_X(n_samples=n_samples, rho=rho, seed=seed)
    betas = np.zeros((n_sim, 2))
    
    for i in range(n_sim):
        epsilon = np.random.normal(0, sigma, size=n_samples)
        y = X @ beta_true + epsilon
        betas[i] = ols_beta(X, y)
    
    return betas, X

if __name__ == "__main__":
    # 实验 A: 正交
    betas_A, X_A = run_simulation(rho=0.0)
    # 实验 B: 高度共线
    betas_B, X_B = run_simulation(rho=0.99)
    
    # 经验协方差矩阵
    cov_empirical_B = np.cov(betas_B, rowvar=False)
    print("Empirical Covariance (B):\n", cov_empirical_B)
    
    # 理论协方差矩阵
    sigma = 2.0
    cov_theoretical_B = sigma**2 * np.linalg.inv(X_B.T @ X_B)
    print("Theoretical Covariance (B):\n", cov_theoretical_B)