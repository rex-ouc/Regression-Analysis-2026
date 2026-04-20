# main.py
import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_X
from simulation import run_simulation, ols_beta

def plot_betas(betas_A, betas_B, beta_true=np.array([5.0, 3.0]), save_path="scatter_plot.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(betas_A[:,0], betas_A[:,1], alpha=0.5, label='Orthogonal (rho=0.0)')
    plt.scatter(betas_B[:,0], betas_B[:,1], alpha=0.5, label='High Collinearity (rho=0.99)')
    plt.scatter(beta_true[0], beta_true[1], color='red', marker='x', s=100, label='True β')
    plt.xlabel('β1 estimates')
    plt.ylabel('β2 estimates')
    plt.title('Monte Carlo β Estimates: Orthogonal vs Collinear Features')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(save_path)
    plt.show()

def main():
    beta_true = np.array([5.0, 3.0])
    sigma = 2.0
    n_samples = 100
    n_sim = 1000

    # 实验 A: 正交特征
    print("Running Simulation A (rho=0.0)...")
    betas_A, X_A = run_simulation(rho=0.0, n_samples=n_samples, n_sim=n_sim, beta_true=beta_true, sigma=sigma)

    # 实验 B: 高度共线
    print("Running Simulation B (rho=0.99)...")
    betas_B, X_B = run_simulation(rho=0.99, n_samples=n_samples, n_sim=n_sim, beta_true=beta_true, sigma=sigma)

    # 经验协方差矩阵
    cov_empirical_B = np.cov(betas_B, rowvar=False)
    print("Empirical Covariance (B):\n", cov_empirical_B)

    # 理论协方差矩阵
    cov_theoretical_B = sigma**2 * np.linalg.inv(X_B.T @ X_B)
    print("Theoretical Covariance (B):\n", cov_theoretical_B)

    # 绘图
    plot_betas(betas_A, betas_B, beta_true=beta_true, save_path="scatter_plot.png")

if __name__ == "__main__":
    main()