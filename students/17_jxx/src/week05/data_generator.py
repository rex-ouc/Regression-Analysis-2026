import numpy as np

def generate_design_matrix(n_samples: int, rho: float) -> np.ndarray:
    """
    构造带有共线性的设计矩阵 X（Fixed Design）
    :param n_samples: 样本数量
    :param rho: X1 和 X2 的相关系数
    :return: n_samples × 2 的设计矩阵
    """
    mean = [0, 0]
    cov = [
        [1, rho],
        [rho, 1]
    ]
    X = np.random.multivariate_normal(mean, cov, size=n_samples)
    return X