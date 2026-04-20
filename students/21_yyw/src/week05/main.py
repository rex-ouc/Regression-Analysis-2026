"""
入口程序：main.py
Week 05 实验：协方差与多重共线性

实验流程：
1. 正交场景 (ρ = 0.0): 蒙特卡洛模拟 1000 次
2. 共线性场景 (ρ = 0.99): 蒙特卡洛模拟 1000 次
3. 打印协方差矩阵对比
4. 生成可视化散点图
"""

import numpy as np
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import run_simulation, print_covariance_matrices
from analysis import plot_beta_distribution, plot_correlation_analysis


def main():
    # 实验参数
    N_SIMULATIONS = 1000    # 模拟次数
    N_SAMPLES = 1000        # 样本量
    TRUE_BETA = np.array([5.0, 3.0])  # 真实参数 [β₁, β₂]
    SIGMA = 2.0             # 噪音标准差
    
    print("=" * 70)
    print("Week 05: 协方差与多重共线性 - 蒙特卡洛模拟")
    print("=" * 70)
    print(f"\n实验配置:")
    print(f"  - 模拟次数: {N_SIMULATIONS}")
    print(f"  - 样本量: {N_SAMPLES}")
    print(f"  - 真实参数: β₁ = {TRUE_BETA[0]}, β₂ = {TRUE_BETA[1]}")
    print(f"  - 噪音标准差: σ = {SIGMA}")
    
    # 创建 assets 目录
    os.makedirs("assets", exist_ok=True)
    
    # ========== 实验 A: 正交场景 (ρ = 0.0) ==========
    print("\n" + "=" * 70)
    print("实验 A: 正交/独立特征 (ρ = 0.0)")
    print("=" * 70)
    
    results_orth, X_orth, cov_emp_orth, cov_theo_orth = run_simulation(
        n_simulations=N_SIMULATIONS,
        n_samples=N_SAMPLES,
        rho=0.0,
        true_beta=TRUE_BETA,
        sigma=SIGMA,
        seed=42
    )
    
    print(f"\n正交场景统计:")
    print(f"  β̂₁ 均值: {results_orth['beta1_hat'].mean():.6f} (理论: {TRUE_BETA[0]})")
    print(f"  β̂₂ 均值: {results_orth['beta2_hat'].mean():.6f} (理论: {TRUE_BETA[1]})")
    print(f"  β̂₁ 方差: {results_orth['beta1_hat'].var():.6f}")
    print(f"  β̂₂ 方差: {results_orth['beta2_hat'].var():.6f}")
    print(f"  Corr(β̂₁, β̂₂): {results_orth['beta1_hat'].corr(results_orth['beta2_hat']):.6f}")
    
    print_covariance_matrices(cov_emp_orth, cov_theo_orth, rho=0.0)
    
    # ========== 实验 B: 高度共线性场景 (ρ = 0.99) ==========
    print("\n" + "=" * 70)
    print("实验 B: 高度共线性 (ρ = 0.99)")
    print("=" * 70)
    
    results_collinear, X_collinear, cov_emp_coll, cov_theo_coll = run_simulation(
        n_simulations=N_SIMULATIONS,
        n_samples=N_SAMPLES,
        rho=0.99,
        true_beta=TRUE_BETA,
        sigma=SIGMA,
        seed=42
    )
    
    print(f"\n共线性场景统计:")
    print(f"  β̂₁ 均值: {results_collinear['beta1_hat'].mean():.6f} (理论: {TRUE_BETA[0]})")
    print(f"  β̂₂ 均值: {results_collinear['beta2_hat'].mean():.6f} (理论: {TRUE_BETA[1]})")
    print(f"  β̂₁ 方差: {results_collinear['beta1_hat'].var():.6f}")
    print(f"  β̂₂ 方差: {results_collinear['beta2_hat'].var():.6f}")
    print(f"  Corr(β̂₁, β̂₂): {results_collinear['beta1_hat'].corr(results_collinear['beta2_hat']):.6f}")
    
    print_covariance_matrices(cov_emp_coll, cov_theo_coll, rho=0.99)
    
    # ========== 可视化 ==========
    print("\n" + "=" * 70)
    print("生成可视化图表...")
    print("=" * 70)
    
    # 散点图：正交 vs 共线
    plot_beta_distribution(results_orth, results_collinear, TRUE_BETA)
    
    # 相关性分析图（可选）
    plot_correlation_analysis(results_orth, results_collinear)
    
    # ========== 总结 ==========
    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)
    print("\n产出文件:")
    print("  - assets/week05_beta_distribution.png (正交 vs 共线散点图)")
    print("  - assets/week05_correlation_analysis.png (相关性分析图)")
    print("\n请查看终端打印的协方差矩阵，并写入 report.md")
    print("=" * 70)


if __name__ == "__main__":
    main()