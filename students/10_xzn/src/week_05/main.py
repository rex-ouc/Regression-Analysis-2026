import numpy as np

results = np.load("simulation_results.npz")
beta_hat_B = results["beta_B"]
X_B = results["X_B"]
TRUE_SIGMA = 2.0


empirical_cov = np.cov(beta_hat_B.T)


xtx = X_B.T @ X_B
xtx_inv = np.linalg.inv(xtx)
theoretical_cov = (TRUE_SIGMA ** 2) * xtx_inv

# --------------------------

print("="*50)
print("实验B: 经验协方差矩阵 (Empirical Covariance)")
print(empirical_cov)
print("\n实验B: 理论协方差矩阵 (Theoretical Covariance)")
print(theoretical_cov)
print("="*50)
print(f"两个矩阵的Frobenius范数差: {np.linalg.norm(empirical_cov - theoretical_cov):.6f}")