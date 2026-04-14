"""
模块：simulation.py
作用：蒙特卡洛模拟，估计参数，对比经验协方差与理论协方差
"""

import numpy as np
import pandas as pd
from data_generator import generate_design_matrix, get_theoretical_covariance, generate_y


def run_simulation(n_simulations, n_samples, rho, true_beta, sigma, seed=42):
    """
    运行蒙特卡洛模拟
    
    Parameters
    ----------
    n_simulations : int
        模拟次数（1000次）
    n_samples : int
        样本量
    rho : float
        相关系数（0.0 或 0.99）
    true_beta : np.ndarray
        真实参数 [β₁, β₂]
    sigma : float
        噪音标准差
    seed : int
        随机种子
    
    Returns
    -------
    results_df : pd.DataFrame
        包含每次模拟的 β̂₁, β̂₂ 的 DataFrame
    X : np.ndarray
        固定的设计矩阵
    cov_theoretical : np.ndarray
        理论协方差矩阵
    """
    # 统计学铁律：只生成一次特征矩阵 X（Fixed Design）
    X = generate_design_matrix(n_samples, rho, seed=seed)
    
    # 初始化随机数生成器（用于生成噪音 ε）
    rng = np.random.default_rng(seed)
    
    # 存储估计结果
    beta_estimates = []
    
    for i in range(n_simulations):
        # 每次只生成新的纯随机噪音 ε
        y = generate_y(X, true_beta, sigma, rng)
        
        # 使用最小二乘法估计参数 β̂ = (XᵀX)⁻¹Xᵀy
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        beta_hat = XtX_inv @ (X.T @ y)
        
        beta_estimates.append(beta_hat)
    
    # 转换为 DataFrame
    results_df = pd.DataFrame(beta_estimates, columns=['beta1_hat', 'beta2_hat'])
    
    # 计算经验协方差矩阵
    cov_empirical = np.cov(results_df['beta1_hat'], results_df['beta2_hat'])
    
    # 计算理论协方差矩阵
    sigma_sq = sigma ** 2
    cov_theoretical = get_theoretical_covariance(X, sigma_sq)
    
    return results_df, X, cov_empirical, cov_theoretical


def print_covariance_matrices(cov_empirical, cov_theoretical, rho):
    """
    打印经验协方差矩阵和理论协方差矩阵
    """
    print("\n" + "=" * 60)
    print(f"协方差矩阵对比 (ρ = {rho})")
    print("=" * 60)
    
    print("\n【经验协方差矩阵 (Empirical)】")
    print(f"Var(β̂₁) = {cov_empirical[0, 0]:.6f}")
    print(f"Var(β̂₂) = {cov_empirical[1, 1]:.6f}")
    print(f"Cov(β̂₁, β̂₂) = {cov_empirical[0, 1]:.6f}")
    print(f"Corr(β̂₁, β̂₂) = {cov_empirical[0, 1] / np.sqrt(cov_empirical[0, 0] * cov_empirical[1, 1]):.6f}")
    
    print("\n【理论协方差矩阵 (Theoretical)】")
    print(f"Var(β̂₁) = {cov_theoretical[0, 0]:.6f}")
    print(f"Var(β̂₂) = {cov_theoretical[1, 1]:.6f}")
    print(f"Cov(β̂₁, β̂₂) = {cov_theoretical[0, 1]:.6f}")
    print(f"Corr(β̂₁, β̂₂) = {cov_theoretical[0, 1] / np.sqrt(cov_theoretical[0, 0] * cov_theoretical[1, 1]):.6f}")
    
    print("\n【矩阵差异】")
    diff = np.abs(cov_empirical - cov_theoretical)
    print(f"最大绝对差异: {np.max(diff):.6e}")