import matplotlib.pyplot as plt
import numpy as np
from simulation import betas_hat_A, betas_hat_B, beta_true
import os

# 自动创建文件夹
os.makedirs('./students/04_lyq/week05/docs', exist_ok=True)

# 创建 1行2列 的子图，两张图并排对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ===================== 左图：实验 A（独立特征 ρ=0.0）=====================
ax1.scatter(
    betas_hat_A[:, 0], betas_hat_A[:, 1],
    alpha=0.6, color='steelblue', label='Experiment A (ρ=0.0)'
)
ax1.scatter(
    beta_true[0], beta_true[1],
    color='black', s=200, marker='*', label='True β = (5, 3)', edgecolors='gold', linewidth=2
)
ax1.set_xlabel(r'$\hat{\beta}_1$', fontsize=14)
ax1.set_ylabel(r'$\hat{\beta}_2$', fontsize=14)
ax1.set_title('Experiment A: Orthogonal Features (ρ=0.0)', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# ===================== 右图：实验 B（高度共线性 ρ=0.99）=====================
ax2.scatter(
    betas_hat_B[:, 0], betas_hat_B[:, 1],
    alpha=0.6, color='crimson', label='Experiment B (ρ=0.99)'
)
ax2.scatter(
    beta_true[0], beta_true[1],
    color='black', s=200, marker='*', label='True β = (5, 3)', edgecolors='gold', linewidth=2
)
ax2.set_xlabel(r'$\hat{\beta}_1$', fontsize=14)
ax2.set_ylabel(r'$\hat{\beta}_2$', fontsize=14)
ax2.set_title('Experiment B: Multicollinearity (ρ=0.99)', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.suptitle('Monte Carlo Simulation: Beta Estimates Comparison', fontsize=16)
plt.tight_layout()

# 保存到指定路径
plt.savefig('./students/04_lyq/week05/docs/covariance_scatter_two_plots.png', dpi=300)
plt.show()
