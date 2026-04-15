import numpy as np

def generate_design_matrix(n_samples: int, rho: float) -> np.ndarray:
    """
    生成包含两个特征X1和X2的设计矩阵，控制相关系数rho
    :param n_samples: 样本量
    :param rho: X1和X2的相关系数，范围[-1, 1]
    :return: 设计矩阵X，shape=(n_samples, 2)
    """
    # 生成标准正态分布的X1
    X1 = np.random.normal(loc=0, scale=1, size=n_samples)
    # 生成与X1相关的X2
    X2 = rho * X1 + np.sqrt(1 - rho**2) * np.random.normal(loc=0, scale=1, size=n_samples)
    # 组合为设计矩阵
    X = np.column_stack([X1, X2])
    return X