"""
数据生成模块
提供生成多元线性回归数据的功能
"""

import numpy as np


def generate_data(n_samples=10000, n_features=10, noise_std=1.0, random_state=42):
    """
    生成多元线性回归数据: y = X @ beta + epsilon

    参数:
        n_samples (int): 样本量 N
        n_features (int): 特征维度 P
        noise_std (float): 噪声标准差
        random_state (int): 随机种子

    返回:
        X (np.ndarray): 特征矩阵 (N, P)
        y (np.ndarray): 目标向量 (N,)
        beta_true (np.ndarray): 真实系数 (P,)
    """
    rng = np.random.default_rng(random_state)

    # 生成特征矩阵（标准正态分布）
    X = rng.normal(0, 1, size=(n_samples, n_features))

    # 生成真实系数（均匀分布，避免过大的值）
    beta_true = rng.uniform(-2, 2, size=n_features)

    # 生成噪声
    epsilon = rng.normal(0, noise_std, size=n_samples)

    # 生成目标变量
    y = X @ beta_true + epsilon

    return X, y, beta_true
