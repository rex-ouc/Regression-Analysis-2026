import matplotlib.pyplot as plt
import numpy as np

def plot_beta_scatter(
    estimates_a: np.ndarray,
    estimates_b: np.ndarray,
    beta_true: np.ndarray = np.array([5.0, 3.0])
):
    """
    绘制正交vs共线性的beta估计散点图
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制实验A（正交）散点
    ax.scatter(
        estimates_a[:, 0], estimates_a[:, 1],
        alpha=0.5, label='Orthogonal Features (ρ=0.0)', color='cornflowerblue'
    )
    # 绘制实验B（共线性）散点
    ax.scatter(
        estimates_b[:, 0], estimates_b[:, 1],
        alpha=0.5, label='High Multicollinearity (ρ=0.99)', color='salmon'
    )
    # 标记真实参数点
    ax.scatter(
        beta_true[0], beta_true[1],
        color='black', marker='*', s=200, label='True Parameter Point'
    )
    
    ax.set_xlabel(r'$\hat{\beta}_1$', fontsize=14)
    ax.set_ylabel(r'$\hat{\beta}_2$', fontsize=14)
    ax.set_title('Orthogonal vs Collinear OLS Estimates', fontsize=16)
    ax.legend()
    plt.tight_layout()
    plt.savefig('beta_scatter.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 模拟数据测试绘图
    np.random.seed(42)
    est_a = np.random.multivariate_normal([5,3], [[0.04,0],[0,0.04]], 1000)
    est_b = np.random.multivariate_normal([5,3], [[40,-39.6],[-39.6,40]], 1000)
    plot_beta_scatter(est_a, est_b)