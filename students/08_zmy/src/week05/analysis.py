"""
分析模块：计算经验/理论协方差矩阵，绘制散点图
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_empirical_covariance(betas):
    """
    根据多次模拟的估计值计算经验协方差矩阵
    参数:
        betas (np.ndarray): 形状 (n_simulations, P)
    返回:
        cov_emp (np.ndarray): 形状 (P, P)
    """
    return np.cov(betas, rowvar=False)

def compute_theoretical_covariance(X, sigma):
    """
    根据理论公式 σ^2 (X^T X)^{-1} 计算理论协方差矩阵
    参数:
        X (np.ndarray): 设计矩阵 (N, P)
        sigma (float): 噪声标准差
    返回:
        cov_theo (np.ndarray): 形状 (P, P)
    """
    XtX_inv = np.linalg.inv(X.T @ X)
    return sigma**2 * XtX_inv

def plot_scatter(betas_orth, betas_collin, true_beta, save_path=None):
    """
    绘制正交特征和共线特征的估计点散点图
    参数:
        betas_orth (np.ndarray): 形状 (n_sim, 2) 正交情形估计值
        betas_collin (np.ndarray): 共线情形估计值
        true_beta (np.ndarray): 真实系数 [beta1, beta2]
        save_path (str): 图片保存路径，若 None 则显示
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(betas_orth[:, 0], betas_orth[:, 1],
                alpha=0.5, label=f'ρ = 0.0 (orthogonal)', s=10)
    plt.scatter(betas_collin[:, 0], betas_collin[:, 1],
                alpha=0.5, label=f'ρ = 0.99 (collinear)', s=10)
    plt.scatter(true_beta[0], true_beta[1], color='red', marker='X', s=200,
                label=f'True β = {true_beta}')
    plt.xlabel('β₁ estimates')
    plt.ylabel('β₂ estimates')
    plt.title('Monte Carlo: Distribution of β Estimates under Different ρ')
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Scatter plot saved to {save_path}")
    else:
        plt.show()

def print_covariance_matrices(cov_theo, cov_emp, rho_label):
    """
    打印理论协方差和经验协方差矩阵
    """
    print(f"\n--- {rho_label} ---")
    print("Theoretical Covariance Matrix (σ² (XᵀX)⁻¹):")
    print(np.round(cov_theo, 6))
    print("Empirical Covariance Matrix (from simulations):")
    print(np.round(cov_emp, 6))