"""
数据生成模块：生成具有指定相关系数 rho 的两个特征 X1, X2
"""
import numpy as np

def generate_design_matrix(n_samples=1000, rho=0.0, random_state=42):
    """
    生成固定设计矩阵 X (N x 2)，其中 X1 和 X2 的相关系数近似为 rho。
    使用 Cholesky 分解生成相关正态变量。

    参数:
        n_samples (int): 样本量
        rho (float): X1 与 X2 的目标相关系数 (-1 <= rho <= 1)
        random_state (int): 随机种子

    返回:
        X (np.ndarray): 形状 (n_samples, 2) 的设计矩阵
    """
    rng = np.random.default_rng(random_state)
    # 生成独立标准正态变量
    Z = rng.normal(0, 1, size=(n_samples, 2))
    # 协方差矩阵 [1, rho; rho, 1] 的 Cholesky 分解
    L = np.linalg.cholesky([[1, rho], [rho, 1]])
    X = Z @ L.T   # 使得 X 的协方差矩阵为 [[1, rho], [rho, 1]]
    return X