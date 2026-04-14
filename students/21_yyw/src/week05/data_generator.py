"""
模块：data_generator.py
作用：生成带有共线性的设计矩阵 X
"""

import numpy as np


def generate_design_matrix(n_samples, rho, seed=42):
    """
    生成包含两个特征 X1 和 X2 的设计矩阵，具有指定的相关系数 rho
    
    Parameters
    ----------
    n_samples : int
        样本量 N
    rho : float
        相关系数 ρ，控制 X1 和 X2 的线性相关程度
        - ρ = 0: 正交/独立
        - ρ = 0.99: 高度共线性
    seed : int
        随机种子（保证可复现）
    
    Returns
    -------
    X : np.ndarray, shape (n_samples, 2)
        设计矩阵，包含 X1 和 X2 两列
    """
    np.random.seed(seed)
    
    # 生成两个标准正态分布的随机变量
    Z1 = np.random.randn(n_samples)
    Z2 = np.random.randn(n_samples)
    
    # 构造相关系数为 rho 的两个特征
    # X1 = Z1
    # X2 = ρ * Z1 + sqrt(1-ρ^2) * Z2
    X1 = Z1
    X2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    # 组合成设计矩阵 (n_samples, 2)
    X = np.column_stack([X1, X2])
    
    return X


def get_theoretical_covariance(X, sigma_sq):
    """
    计算理论协方差矩阵: σ² (XᵀX)⁻¹
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 2)
        设计矩阵
    sigma_sq : float
        噪音方差 σ²
    
    Returns
    -------
    cov_theoretical : np.ndarray, shape (2, 2)
        理论协方差矩阵
    """
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    cov_theoretical = sigma_sq * XtX_inv
    return cov_theoretical


def generate_y(X, true_beta, sigma, rng):
    """
    生成目标变量 y = Xβ + ε
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 2)
        设计矩阵（固定）
    true_beta : np.ndarray, shape (2,)
        真实参数 [β₁, β₂]
    sigma : float
        噪音标准差
    rng : np.random.Generator
        随机数生成器对象
    
    Returns
    -------
    y : np.ndarray, shape (n_samples,)
        目标向量
    """
    epsilon = rng.normal(0, sigma, size=X.shape[0])
    y = X @ true_beta + epsilon
    return y