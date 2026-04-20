import numpy as np

def generate_design_matrix(n_samples: int, rho: float) -> np.ndarray:
    # 生成服从标准正态分布的独立随机变量
    z1 = np.random.normal(0, 1, n_samples)
    z2 = np.random.normal(0, 1, n_samples)

    # 构造相关系数为 rho 的 X1, X2
    X1 = z1
    X2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2

    # 拼接为设计矩阵 (n_samples, 2)
    X = np.column_stack((X1, X2))
    return X
