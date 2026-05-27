import numpy as np

def generate_design_matrix(n_samples: int, rho: float) -> np.ndarray:
    """
    生成包含两个特征X1和X2的设计矩阵，控制相关系数rho
    :param n_samples: 样本量
    :param rho: X1和X2的相关系数，范围[-1, 1]
    :return: 设计矩阵X，shape=(n_samples, 2)
    """
    X1 = np.random.normal(loc=0, scale=1, size=n_samples)
    X2 = rho * X1 + np.sqrt(1 - rho**2) * np.random.normal(loc=0, scale=1, size=n_samples)
    X = np.column_stack([X1, X2])
    return X