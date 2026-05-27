"""
第五周作业主程序：对比正交特征与高度共线性特征下的协方差矩阵和估计分布
"""
import os
import numpy as np
from data_generator import generate_design_matrix
from simulation import run_monte_carlo
from analysis import (
    compute_empirical_covariance,
    compute_theoretical_covariance,
    plot_scatter,
    print_covariance_matrices
)

def main():
    # 实验参数
    n_samples = 1000          # 样本量 N
    n_simulations = 1000      # 蒙特卡洛次数
    beta_true = np.array([5.0, 3.0])   # 真实系数 [β1, β2]
    sigma = 2.0               # 噪声标准差

    # 实验 A: 正交特征 (ρ = 0.0)
    print("=== Experiment A: Orthogonal Features (ρ = 0.0) ===")
    X_orth = generate_design_matrix(n_samples, rho=0.0, random_state=42)
    betas_orth = run_monte_carlo(X_orth, beta_true, sigma, n_simulations)
    cov_theo_orth = compute_theoretical_covariance(X_orth, sigma)
    cov_emp_orth = compute_empirical_covariance(betas_orth)
    print_covariance_matrices(cov_theo_orth, cov_emp_orth, "ρ = 0.0")

    # 实验 B: 高度共线性 (ρ = 0.99)
    print("\n=== Experiment B: High Collinearity (ρ = 0.99) ===")
    X_collin = generate_design_matrix(n_samples, rho=0.99, random_state=42)
    betas_collin = run_monte_carlo(X_collin, beta_true, sigma, n_simulations)
    cov_theo_collin = compute_theoretical_covariance(X_collin, sigma)
    cov_emp_collin = compute_empirical_covariance(betas_collin)
    print_covariance_matrices(cov_theo_collin, cov_emp_collin, "ρ = 0.99")

    # 绘制散点图对比
    save_path = "students/08_zmy/src/week05/assets/week05_scatter.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_scatter(betas_orth, betas_collin, beta_true, save_path=save_path)

    print("\n实验完成！散点图已保存至 students/08_zmy/src/week05/assets/week05_scatter.png")

if __name__ == "__main__":
    main()