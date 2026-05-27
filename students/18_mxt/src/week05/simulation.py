import numpy as np
from data_generator import generate_design_matrix
from solvers import AnalyticalSolver

def monte_carlo_simulation(
    n_samples: int = 1000,
    rho: float = 0.0,
    n_simulations: int = 1000,
    beta_true: np.ndarray = np.array([5.0, 3.0]),
    sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    蒙特卡洛模拟，执行n_simulations次线性回归拟合，记录估计值
    :return: (估计值数组, 固定设计矩阵X)
    """
    # 固定设计矩阵（循环外只生成一次）
    X = generate_design_matrix(n_samples, rho)
    beta_hat_list = []
    solver = AnalyticalSolver()

    for _ in range(n_simulations):
        # 每次生成新的随机噪声
        epsilon = np.random.normal(loc=0, scale=sigma, size=n_samples)
        # 生成响应变量y
        y = X @ beta_true + epsilon
        # 拟合模型，记录估计值（跳过截距项）
        solver.fit(X, y)
        beta_hat_list.append(solver.beta[1:])

    return np.array(beta_hat_list), X