"""
模块：simulation.py
作用：定义数值试验的核心组件 (DGP, 估计, 循环, 分析)
注意：本文件被设计为被导入的模块，不要在这里直接运行业务逻辑。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# ==========================================
# 1. Generate Data (数据生成过程 DGP)
# ==========================================
def generate_data(n, true_params, noise_std, rng):
    """
    生成线性回归模拟数据
    
    设计逻辑：将"上帝视角"独立封装。未来可以加入异方差、多重共线性等"毒药"。
    
    Parameters
    ----------
    n : int
        样本量
    true_params : list or np.ndarray
        真实参数，第一个是截距，后面是斜率系数
    noise_std : float
        噪音强度（标准差）
    rng : np.random.Generator
        随机数生成器对象（保证随机流安全且可复现）
    
    Returns
    -------
    X : np.ndarray
        特征矩阵，shape (n, p-1)
    y : np.ndarray
        目标向量，shape (n,)
    """
    true_params = np.array(true_params)
    p = len(true_params)  # 参数个数（含截距）
    n_features = p - 1    # 特征个数（不含截距）
    
    # 生成特征矩阵 X
    X = rng.uniform(0, 10, size=(n, n_features))
    
    # 生成噪音 ε ~ N(0, noise_std^2)
    epsilon = rng.normal(0, noise_std, n)
    
    # 添加截距列（用于计算 y）
    X_with_const = np.column_stack([np.ones(n), X])
    
    # y = X * β + ε
    y = X_with_const @ true_params + epsilon
    
    return X, y


# ==========================================
# 2. Estimate Once (单次拟合与评估)
# ==========================================
def estimate_once(X, y):
    """
    单次拟合，返回各种估计指标
    
    设计逻辑：黑盒化模型接口。并排调用 Statsmodels 和 Sklearn，
    直观对比"推断API"与"预测API"的差异。
    
    Parameters
    ----------
    X : np.ndarray
        特征矩阵，shape (n, p-1)
    y : np.ndarray
        目标向量，shape (n,)
    
    Returns
    -------
    dict
        包含各类指标的扁平字典（便于后期解析）
    """
    X_with_const = sm.add_constant(X)
    
    # ========== statsmodels（推断API） ==========
    stats_model = sm.OLS(y, X_with_const).fit()
    ols_beta_0 = stats_model.params[0]
    ols_beta_1 = stats_model.params[1]
    p_value_0 = stats_model.pvalues[0]
    p_value_1 = stats_model.pvalues[1]
    
    # ========== sklearn（预测API） ==========
    sklearn_model = LinearRegression()
    sklearn_model.fit(X, y)
    sklearn_beta_0 = sklearn_model.intercept_
    sklearn_beta_1 = sklearn_model.coef_[0]
    r2 = sklearn_model.score(X, y)
    
    # ========== 公式法（手动验证，可选） ==========
    x_mean, y_mean = np.mean(X[:, 0]), np.mean(y)
    numerator = np.sum((X[:, 0] - x_mean) * (y - y_mean))
    denominator = np.sum((X[:, 0] - x_mean) ** 2)
    formula_beta_1 = numerator / denominator
    formula_beta_0 = y_mean - formula_beta_1 * x_mean
    
    return {
        # statsmodels 结果
        "ols_beta_0": ols_beta_0,
        "ols_beta_1": ols_beta_1,
        "p_value_0": p_value_0,
        "p_value_1": p_value_1,
        # sklearn 结果
        "sklearn_beta_0": sklearn_beta_0,
        "sklearn_beta_1": sklearn_beta_1,
        "r2": r2,
        # 公式法结果（验证用）
        "formula_beta_0": formula_beta_0,
        "formula_beta_1": formula_beta_1,
    }


# ==========================================
# 3. Loop (蒙特卡洛循环)
# ==========================================
def loop(n_simulations, n, true_params, noise_std, seed=42):
    """
    蒙特卡洛模拟循环
    
    设计逻辑：连接概率论与计算的桥梁。模拟估计量的分布性质。
    
    注意事项：
    1. 在这一层初始化核心的随机数生成器对象 (rng)
    2. 性能优化：用 list 收集结果，循环结束后一次性转 DataFrame
    
    Parameters
    ----------
    n_simulations : int
        模拟次数
    n : int
        每次模拟的样本量
    true_params : list or np.ndarray
        真实参数
    noise_std : float
        噪音强度
    seed : int
        随机种子，保证可复现
    
    Returns
    -------
    pd.DataFrame
        包含所有模拟结果的DataFrame
    """
    true_params = np.array(true_params)
    
    # 1. 实例化随机数生成器
    rng = np.random.default_rng(seed)
    
    # 2. 初始化空列表（性能优化：不用 df.append）
    results_list = []
    
    # 3. 蒙特卡洛循环
    for i in range(n_simulations):
        # a. 生成数据
        X, y = generate_data(n, true_params, noise_std, rng)
        
        # b. 单次估计
        metrics_dict = estimate_once(X, y)
        
        # 添加模拟次数标识
        metrics_dict["simulation_id"] = i
        
        # c. 收集结果
        results_list.append(metrics_dict)
    
    # 4. 一次性转换为DataFrame
    df_results = pd.DataFrame(results_list)
    
    return df_results


# ==========================================
# 4. Analysis (分析与报告物料生成)
# ==========================================
def analysis(df_results, true_params):
    """
    分析模拟结果，生成报告材料
    
    设计逻辑：表现层分离。只负责计算 Bias、Variance，并输出人类可读的材料。
    
    注意事项：
    - 严禁使用 plt.show()，必须用 plt.savefig()
    - 打印 Markdown 表格，便于直接复制到报告
    
    Parameters
    ----------
    df_results : pd.DataFrame
        loop() 返回的模拟结果DataFrame
    true_params : list or np.ndarray
        真实参数 [β₀, β₁]
    
    Returns
    -------
    None
        产生文件副作用：保存图片，打印表格
    """
    true_params = np.array(true_params)
    beta_0_true, beta_1_true = true_params[0], true_params[1]
    
    # ========== 1. 计算偏差 (Bias) ==========
    bias_ols_0 = df_results["ols_beta_0"].mean() - beta_0_true
    bias_ols_1 = df_results["ols_beta_1"].mean() - beta_1_true
    bias_sklearn_0 = df_results["sklearn_beta_0"].mean() - beta_0_true
    bias_sklearn_1 = df_results["sklearn_beta_1"].mean() - beta_1_true
    
    # ========== 2. 计算方差 (Variance) ==========
    var_ols_0 = df_results["ols_beta_0"].var()
    var_ols_1 = df_results["ols_beta_1"].var()
    var_sklearn_0 = df_results["sklearn_beta_0"].var()
    var_sklearn_1 = df_results["sklearn_beta_1"].var()
    
    # ========== 3. 计算均方误差 (MSE) ==========
    mse_ols_0 = ((df_results["ols_beta_0"] - beta_0_true) ** 2).mean()
    mse_ols_1 = ((df_results["ols_beta_1"] - beta_1_true) ** 2).mean()
    mse_sklearn_0 = ((df_results["sklearn_beta_0"] - beta_0_true) ** 2).mean()
    mse_sklearn_1 = ((df_results["sklearn_beta_1"] - beta_1_true) ** 2).mean()
    
    # ========== 4. 打印 Markdown 表格 ==========
    print("\n" + "=" * 70)
    print(f"蒙特卡洛模拟结果（基于 {len(df_results)} 次模拟）")
    print(f"真实参数：β₀ = {beta_0_true}, β₁ = {beta_1_true}")
    print("=" * 70)
    
    print("\n### 参数估计偏差 (Bias)")
    print("| 方法 | β₀ 偏差 | β₁ 偏差 |")
    print("|------|---------|---------|")
    print(f"| statsmodels | {bias_ols_0:.6f} | {bias_ols_1:.6f} |")
    print(f"| sklearn | {bias_sklearn_0:.6f} | {bias_sklearn_1:.6f} |")
    
    print("\n### 参数估计方差 (Variance)")
    print("| 方法 | Var(β̂₀) | Var(β̂₁) |")
    print("|------|---------|---------|")
    print(f"| statsmodels | {var_ols_0:.6f} | {var_ols_1:.6f} |")
    print(f"| sklearn | {var_sklearn_0:.6f} | {var_sklearn_1:.6f} |")
    
    print("\n### 均方误差 (MSE)")
    print("| 方法 | MSE(β̂₀) | MSE(β̂₁) |")
    print("|------|---------|---------|")
    print(f"| statsmodels | {mse_ols_0:.6f} | {mse_ols_1:.6f} |")
    print(f"| sklearn | {mse_sklearn_0:.6f} | {mse_sklearn_1:.6f} |")
    
    # 验证两种方法一致性
    diff_0_mean = (df_results["ols_beta_0"] - df_results["sklearn_beta_0"]).mean()
    diff_1_mean = (df_results["ols_beta_1"] - df_results["sklearn_beta_1"]).mean()
    print("\n### 方法一致性验证")
    print(f"statsmodels 与 sklearn β₀ 估计差异均值: {diff_0_mean:.10f}")
    print(f"statsmodels 与 sklearn β₁ 估计差异均值: {diff_1_mean:.10f}")
    
    # ========== 5. 绘制分布图 ==========
    # 创建 assets 目录（如果不存在）
    import os
    os.makedirs("assets", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：β₀ 的分布
    axes[0].hist(df_results["ols_beta_0"], bins=30, alpha=0.7, 
                 label="statsmodels", density=True)
    axes[0].hist(df_results["sklearn_beta_0"], bins=30, alpha=0.5, 
                 label="sklearn", density=True)
    axes[0].axvline(beta_0_true, color='r', linestyle='--', linewidth=2,
                    label=f'真实值 β₀={beta_0_true}')
    axes[0].set_xlabel("β₀ 估计值")
    axes[0].set_ylabel("密度")
    axes[0].set_title("β₀ 估计量的分布")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 右图：β₁ 的分布
    axes[1].hist(df_results["ols_beta_1"], bins=30, alpha=0.7, 
                 label="statsmodels", density=True)
    axes[1].hist(df_results["sklearn_beta_1"], bins=30, alpha=0.5, 
                 label="sklearn", density=True)
    axes[1].axvline(beta_1_true, color='r', linestyle='--', linewidth=2,
                    label=f'真实值 β₁={beta_1_true}')
    axes[1].set_xlabel("β₁ 估计值")
    axes[1].set_ylabel("密度")
    axes[1].set_title("β₁ 估计量的分布")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片（不用 plt.show()）
    plt.savefig("assets/beta_dist.png", dpi=150, bbox_inches='tight')
    print("\n✅ 图片已保存至: assets/beta_dist.png")
    
    # 关闭图形释放内存
    plt.close()