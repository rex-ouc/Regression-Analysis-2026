# data_generator.py
import numpy as np

def generate_X(n_samples=100, rho=0.0, seed=None):
    """
    生成带相关性的设计矩阵 X (n_samples x 2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    mean = [0, 0]
    cov = [[1, rho],
           [rho, 1]]  # 协方差矩阵
    X = np.random.multivariate_normal(mean, cov, size=n_samples)
    
    return X