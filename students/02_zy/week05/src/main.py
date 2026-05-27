"""
Week05
Author:zhouying
"""
import numpy as np
from data_generator import generate_design_matrix
from simulation import run_monte_carlo, compare_covariance_matrices
from analysis import plot_beta_scatter

# 全局配置
N_SAMPLES = 1000
N_SIMULATIONS = 1000
BETA_TRUE = np.array([5.0, 3.0])
SIGMA_TRUE = 2.0
SEED = 42

if __name__ == "__main__":
    print("=== 第5周作业:共线性与协方差模拟 ===")

    # 1. 运行两组实验
    print("\n[1/4] 运行实验A: ρ=0.0(正交特征)")
    estimates_a = run_monte_carlo(
        rho=0.0,
        n_simulations=N_SIMULATIONS,
        n_samples=N_SAMPLES,
        beta_true=BETA_TRUE,
        sigma_true=SIGMA_TRUE,
        seed=SEED
    )

    print("[2/4] 运行实验B: ρ=0.99(高度共线性)")
    estimates_b = run_monte_carlo(
        rho=0.99,
        n_simulations=N_SIMULATIONS,
        n_samples=N_SAMPLES,
        beta_true=BETA_TRUE,
        sigma_true=SIGMA_TRUE,
        seed=SEED
    )

    # 2. 对比实验B的协方差矩阵
    print("\n[3/4] 对比实验B的协方差矩阵")
    X_b = generate_design_matrix(n_samples=N_SAMPLES, rho=0.99, seed=SEED)
    emp_cov, theo_cov = compare_covariance_matrices(estimates_b, X_b, SIGMA_TRUE)

    print("=== 经验协方差矩阵 ===")
    print(emp_cov.round(4))
    print("\n=== 理论协方差矩阵 ===")
    print(theo_cov.round(4))

    # 3. 绘制散点图
    print("\n[4/4] 绘制散点图,已保存为beta_scatter.png")
    plot_beta_scatter(estimates_a, estimates_b)