import numpy as np
from data_generator import generate_design_matrix

TRUE_BETA = np.array([5.0, 3.0])
TRUE_SIGMA = 2.0
N_SIMULATIONS = 1000
N_SAMPLES = 100  

def run_monte_carlo(rho: float) -> np.ndarray:

    X = generate_design_matrix(n_samples=N_SAMPLES, rho=rho)
    xtx_inv = np.linalg.inv(X.T @ X)
    
    beta_hat_records = np.zeros((N_SIMULATIONS, 2))
    
    for i in range(N_SIMULATIONS):
        epsilon = np.random.normal(loc=0, scale=TRUE_SIGMA, size=N_SAMPLES)
        y = X @ TRUE_BETA + epsilon
        beta_hat = xtx_inv @ X.T @ y
        beta_hat_records[i] = beta_hat
    
    return beta_hat_records

#两组对比实验
if __name__ == "__main__":
    # 实验A: 正交特征
    beta_hat_A = run_monte_carlo(rho=0.0)
    # 实验B: 高度共线性
    beta_hat_B = run_monte_carlo(rho=0.99)
    
    # 给T34用
    np.savez("simulation_results.npz", 
             beta_A=beta_hat_A, 
             beta_B=beta_hat_B,
             X_B=generate_design_matrix(N_SAMPLES, 0.99))  
    print("模拟完成，结果已保存为simulation_results.npz")