"""
蒙特卡洛模拟模块：对固定设计矩阵 X 重复生成噪声并估计系数
"""
import numpy as np

def ols_estimate(X, y):
    """
    最小二乘估计 (无截距)
    参数:
        X (np.ndarray): (N, P) 设计矩阵
        y (np.ndarray): (N,) 响应变量
    返回:
        beta_hat (np.ndarray): 估计系数 (P,)
    """
    # 求解正规方程 (X^T X) beta = X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)

def run_monte_carlo(X, beta_true, sigma, n_simulations=1000):
    """
    执行蒙特卡洛模拟

    参数:
        X (np.ndarray): 固定设计矩阵 (N, P)
        beta_true (np.ndarray): 真实系数 (P,)
        sigma (float): 噪声标准差
        n_simulations (int): 模拟次数

    返回:
        betas (np.ndarray): 形状 (n_simulations, P) 所有估计值
    """
    N = X.shape[0]
    rng = np.random.default_rng(seed=42)   # 保证噪声可复现（但每次模拟独立）
    betas = []

    for _ in range(n_simulations):
        # 生成新的随机噪声
        epsilon = rng.normal(0, sigma, size=N)
        y = X @ beta_true + epsilon
        beta_hat = ols_estimate(X, y)
        betas.append(beta_hat)

    return np.array(betas)