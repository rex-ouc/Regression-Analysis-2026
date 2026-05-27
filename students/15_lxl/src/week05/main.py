import numpy as np
import os
from simulation import run_monte_carlo, compare_covariance_matrices
from analysis import plot_beta_scatter

def main():
    # 设置参数
    beta_true = np.array([5.0, 3.0])
    sigma_true = 2.0
    n_simulations = 1000
    n_samples = 1000
    
    # 获取当前脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'docs', 'beta_scatter.png')
    
    print("=" * 60)
    print("Week 05 Assignment: Covariance & Multicollinearity")
    print("=" * 60)
    
    # 实验 A: 正交特征 (rho = 0.0)
    print("\n[实验 A] 正交特征 (rho = 0.0)")
    estimates_a, X_a = run_monte_carlo(
        rho=0.0,
        n_simulations=n_simulations,
        n_samples=n_samples,
        beta_true=beta_true,
        sigma_true=sigma_true
    )
    print(f"完成 {n_simulations} 次模拟")
    print(f"估计均值: beta1={estimates_a[:,0].mean():.4f}, beta2={estimates_a[:,1].mean():.4f}")
    print(f"估计标准差: beta1={estimates_a[:,0].std():.4f}, beta2={estimates_a[:,1].std():.4f}")
    
    # 实验 B: 高度共线性 (rho = 0.99)
    print("\n[实验 B] 高度共线性 (rho = 0.99)")
    estimates_b, X_b = run_monte_carlo(
        rho=0.99,
        n_simulations=n_simulations,
        n_samples=n_samples,
        beta_true=beta_true,
        sigma_true=sigma_true
    )
    print(f"完成 {n_simulations} 次模拟")
    print(f"估计均值: beta1={estimates_b[:,0].mean():.4f}, beta2={estimates_b[:,1].mean():.4f}")
    print(f"估计标准差: beta1={estimates_b[:,0].std():.4f}, beta2={estimates_b[:,1].std():.4f}")
    
    # 协方差矩阵对比（实验 B）
    print("\n" + "=" * 60)
    print("[协方差矩阵对比] 实验 B (rho = 0.99)")
    print("=" * 60)
    empirical_cov, theoretical_cov = compare_covariance_matrices(
        estimates_b, X_b, sigma_true
    )
    
    print("\n经验协方差矩阵 (Empirical Covariance Matrix):")
    print(empirical_cov.round(6))
    
    print("\n理论协方差矩阵 (Theoretical Covariance Matrix):")
    print(theoretical_cov.round(6))
    
    print("\n差异矩阵 (Difference):")
    print((empirical_cov - theoretical_cov).round(6))
    
    # 绘制散点图
    print("\n" + "=" * 60)
    print("生成可视化散点图...")
    print("=" * 60)
    plot_beta_scatter(
        estimates_a, estimates_b, beta_true,
        save_path=save_path
    )
    print(f"散点图已保存到 {save_path}")
    
    print("\n" + "=" * 60)
    print("作业完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()