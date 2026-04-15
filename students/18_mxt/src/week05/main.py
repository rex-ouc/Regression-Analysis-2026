import numpy as np
from simulation import monte_carlo_simulation
from analysis import calculate_covariance_matrices, plot_covariance_scatter


def main():
    # 实验参数（完全符合作业要求）
    n_samples = 1000
    n_simulations = 1000
    beta_true = np.array([5.0, 3.0])
    sigma = 2.0

    # ====================== Task2：两组对比实验 ======================
    # 实验A：正交/独立特征 (ρ=0.0)
    print("=" * 70)
    print("【实验A：正交特征 ρ=0.0】")
    beta_hat_A, X_A = monte_carlo_simulation(
        n_samples=n_samples,
        rho=0.0,
        n_simulations=n_simulations,
        beta_true=beta_true,
        sigma=sigma,
    )

    # 实验B：高度共线性 (ρ=0.99)
    print("\n【实验B：高度共线性 ρ=0.99】")
    beta_hat_B, X_B = monte_carlo_simulation(
        n_samples=n_samples,
        rho=0.99,
        n_simulations=n_simulations,
        beta_true=beta_true,
        sigma=sigma,
    )

    # ====================== Task3：理论vs经验协方差矩阵 ======================
    print("\n" + "=" * 70)
    print("【实验A：协方差矩阵对比】")
    emp_cov_A, theo_cov_A = calculate_covariance_matrices(X_A, beta_hat_A, sigma)
    print("经验协方差矩阵：")
    print(np.round(emp_cov_A, 4))
    print("\n理论协方差矩阵：")
    print(np.round(theo_cov_A, 4))

    print("\n" + "=" * 70)
    print("【实验B：协方差矩阵对比】")
    emp_cov_B, theo_cov_B = calculate_covariance_matrices(X_B, beta_hat_B, sigma)
    print("经验协方差矩阵：")
    print(np.round(emp_cov_B, 4))
    print("\n理论协方差矩阵：")
    print(np.round(theo_cov_B, 4))

    # ====================== Task4：协方差散点图 ======================
    print("\n" + "=" * 70)
    print("【绘制协方差散点图】")
    plot_covariance_scatter(beta_hat_A, beta_hat_B, beta_true)
    print("散点图已保存为 covariance_scatter.png")


if __name__ == "__main__":
    main()