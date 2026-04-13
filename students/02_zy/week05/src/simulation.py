import numpy as np
from data_generator import generate_design_matrix

def run_monte_carlo(
    rho: float,
    n_simulations: int = 1000,
    n_samples: int = 1000,
    beta_true: np.ndarray = np.array([5.0, 3.0]),
    sigma_true: float = 2.0,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    执行蒙特卡洛模拟,返回所有估计的beta值
    
    参数:
        rho: X1和X2的相关系数
        n_simulations: 模拟次数
        n_samples: 每次模拟的样本数
        beta_true: 真实参数 [beta1, beta2]
        sigma_true: 噪声标准差
        seed: 随机种子
    返回:
        beta_estimates: (n_simulations, 2) 的估计结果矩阵
    """
    np.random.seed(seed)
    # 1. 生成固定的设计矩阵（Fixed Design）
    X = generate_design_matrix(n_samples=n_samples, rho=rho, seed=seed)
    # 2. 预计算X的伪逆，加速模拟
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_estimates = []
    
    for _ in range(n_simulations):
        # 生成新的噪声和y
        epsilon = np.random.normal(0, sigma_true, n_samples)
        y = X @ np.concatenate([[0], beta_true]) + epsilon  # 常数项系数为0
        
        # OLS估计
        beta_hat = XtX_inv @ X.T @ y
        beta_estimates.append(beta_hat[1:])  # 只保留beta1和beta2
    
    return np.array(beta_estimates)

def compare_covariance_matrices(
    beta_estimates: np.ndarray,
    X: np.ndarray,
    sigma_true: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算并对比经验协方差矩阵和理论协方差矩阵
    """
    # 经验协方差矩阵（用估计结果计算）
    empirical_cov = np.cov(beta_estimates.T)
    # 理论协方差矩阵: σ²(XᵀX)⁻¹，只取beta1和beta2的部分
    XtX_inv = np.linalg.inv(X.T @ X)
    theoretical_cov = sigma_true**2 * XtX_inv[1:, 1:]
    
    return empirical_cov, theoretical_cov

if __name__ == "__main__":
    # 测试高共线性情况
    rho_test = 0.99
    beta_true = np.array([5.0, 3.0])
    sigma_true = 2.0
    
    estimates = run_monte_carlo(rho=rho_test)
    X_fixed = generate_design_matrix(rho=rho_test)
    emp_cov, theo_cov = compare_covariance_matrices(estimates, X_fixed, sigma_true)
    
    print("=== 经验协方差矩阵 ===")
    print(emp_cov.round(4))
    print("\n=== 理论协方差矩阵 ===")
    print(theo_cov.round(4))