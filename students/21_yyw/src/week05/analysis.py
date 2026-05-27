"""
模块：analysis.py
作用：可视化参数估计值的分布
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def setup_chinese_font():
    """设置中文字体"""
    try:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except:
        return False


def plot_beta_distribution(results_orth, results_collinear, true_beta):
    """
    绘制正交场景和共线性场景的参数估计散点图
    
    Parameters
    ----------
    results_orth : pd.DataFrame
        正交场景 (ρ=0) 的估计结果，包含 beta1_hat, beta2_hat
    results_collinear : pd.DataFrame
        共线性场景 (ρ=0.99) 的估计结果
    true_beta : np.ndarray
        真实参数 [β₁, β₂]
    """
    setup_chinese_font()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制正交场景（蓝色，圆形分布）
    ax.scatter(
        results_orth['beta1_hat'], 
        results_orth['beta2_hat'],
        alpha=0.5, s=10, c='steelblue', 
        label=f'正交场景 (ρ = 0.0)'
    )
    
    # 绘制共线性场景（红色，倾斜椭圆分布）
    ax.scatter(
        results_collinear['beta1_hat'], 
        results_collinear['beta2_hat'],
        alpha=0.5, s=10, c='coral', 
        label=f'高度共线性 (ρ = 0.99)'
    )
    
    # 标出真实的 β 点
    ax.scatter(
        true_beta[0], true_beta[1], 
        c='green', s=200, marker='*', 
        edgecolors='black', linewidth=1.5,
        label=f'真实值 β = ({true_beta[0]}, {true_beta[1]})'
    )
    
    # 添加辅助线
    ax.axhline(true_beta[1], color='gray', linestyle='--', alpha=0.5)
    ax.axvline(true_beta[0], color='gray', linestyle='--', alpha=0.5)
    
    # 设置标签和标题
    ax.set_xlabel(r'$\hat{\beta}_1$', fontsize=12)
    ax.set_ylabel(r'$\hat{\beta}_2$', fontsize=12)
    ax.set_title('参数估计量分布：正交 vs 高度共线性', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 设置等比例坐标轴
    ax.set_aspect('equal')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('assets/week05_beta_distribution.png', dpi=150, bbox_inches='tight')
    print("\n✅ 散点图已保存至: assets/week05_beta_distribution.png")
    plt.close()


def plot_correlation_analysis(results_orth, results_collinear):
    """
    绘制相关性分析图（可选，用于深入理解）
    """
    setup_chinese_font()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：正交场景
    ax1 = axes[0]
    ax1.scatter(results_orth['beta1_hat'], results_orth['beta2_hat'], 
                alpha=0.5, s=10, c='steelblue')
    ax1.set_xlabel(r'$\hat{\beta}_1$')
    ax1.set_ylabel(r'$\hat{\beta}_2$')
    ax1.set_title(f'正交场景 (ρ = 0.0)\nCorr = {results_orth["beta1_hat"].corr(results_orth["beta2_hat"]):.4f}')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 右图：共线性场景
    ax2 = axes[1]
    ax2.scatter(results_collinear['beta1_hat'], results_collinear['beta2_hat'], 
                alpha=0.5, s=10, c='coral')
    ax2.set_xlabel(r'$\hat{\beta}_1$')
    ax2.set_ylabel(r'$\hat{\beta}_2$')
    ax2.set_title(f'高度共线性 (ρ = 0.99)\nCorr = {results_collinear["beta1_hat"].corr(results_collinear["beta2_hat"]):.4f}')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('assets/week05_correlation_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ 相关性分析图已保存至: assets/week05_correlation_analysis.png")
    plt.close()