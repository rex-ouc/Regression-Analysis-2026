"""
模块:components.py
作用:定义数值试验的核心组件 (DGP, 估计, 循环, 分析)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ==========================================
# 1. Generate Data (数据生成过程 DGP)
# ==========================================
def generate_data(sample_size, true_params, noise_std, rng) -> tuple:
    """
    生成一元线性回归数据: y = beta_0 + beta_1 * X + epsilon
    其中 epsilon ~ N(0, noise_std^2)

    参数:
        sample_size (int): 样本量
        true_params (list): [beta_0, beta_1]
        noise_std (float): 噪音标准差
        rng (np.random.Generator): 随机数生成器

    返回:
        X (np.ndarray): 特征矩阵 (shape: n, 1)
        y (np.ndarray): 目标向量 (shape: n,)
    """
    # 生成 X ~ Uniform(0, 10)
    X = rng.uniform(0, 10, size=sample_size)

    # 生成噪音 epsilon
    epsilon = rng.normal(0, noise_std, size=sample_size)

    # 计算 y
    beta_0, beta_1 = true_params
    y = beta_0 + beta_1 * X + epsilon

    # sklearn 需要 X 是二维数组
    X = X.reshape(-1, 1)

    return X, y


# ==========================================
# 2. Estimate Once (单次拟合与评估)
# ==========================================
def estimate_once(X, y) -> dict:
    """
    对给定的 X, y 进行三种方法估计，并返回关键指标

    参数:
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 目标向量

    返回:
        dict: 包含所有估计结果和统计指标
    """
    # 手动计算 OLS (公式法)
    n = len(y)
    X_flat = X.flatten()  # 一维便于计算
    X_mean = np.mean(X_flat)
    y_mean = np.mean(y)

    Sxx = np.sum((X_flat - X_mean) ** 2)
    Sxy = np.sum((X_flat - X_mean) * (y - y_mean))

    beta_1_manual = Sxy / Sxx
    beta_0_manual = y_mean - beta_1_manual * X_mean

    # 残差方差和 beta_1 方差
    residuals = y - (beta_0_manual + beta_1_manual * X_flat)
    sigma_sq = np.sum(residuals ** 2) / (n - 2)
    var_beta_1_manual = sigma_sq / Sxx

    # sklearn 线性回归
    sk_model = LinearRegression()
    sk_model.fit(X, y)
    beta_0_sk = sk_model.intercept_
    beta_1_sk = sk_model.coef_[0]
    r2_sk = sk_model.score(X, y)

    # statsmodels OLS (使用数组 API)
    X_sm = sm.add_constant(X)          # 添加截距项
    sm_model = sm.OLS(y, X_sm).fit()
    beta_0_sm = sm_model.params[0]
    beta_1_sm = sm_model.params[1]
    # 提取假设检验的 p-value 和 F 检验
    p_value_beta1 = sm_model.pvalues[1]      # beta_1 的 p-value
    f_statistic = sm_model.fvalue            # 整体 F 统计量
    f_pvalue = sm_model.f_pvalue             # 整体 F 检验 p-value

    # 打包结果字典
    results = {
        "beta_0_manual": beta_0_manual,
        "beta_1_manual": beta_1_manual,
        "var_beta_1_manual": var_beta_1_manual,
        "beta_0_sklearn": beta_0_sk,
        "beta_1_sklearn": beta_1_sk,
        "r2_sklearn": r2_sk,
        "beta_0_statsmodels": beta_0_sm,
        "beta_1_statsmodels": beta_1_sm,
        "p_value_beta1": p_value_beta1,
        "f_statistic": f_statistic,
        "f_pvalue": f_pvalue,
    }
    return results


# ==========================================
# 3. Loop (蒙特卡洛循环)
# ==========================================
def loop(num_simulations, sample_size, true_params, noise_std) -> pd.DataFrame:
    """
    执行蒙特卡洛模拟，重复生成数据并估计

    参数:
        num_simulations (int): 模拟次数
        sample_size (int): 每次模拟的样本量
        true_params (list): [beta_0, beta_1]
        noise_std (float): 噪音标准差

    返回:
        pd.DataFrame: 包含每次模拟所有指标的 DataFrame
    """
    # 初始化随机数生成器 (固定种子，保证可复现)
    rng = np.random.default_rng(seed=42)

    results_list = []

    for i in range(num_simulations):
        # 生成数据
        X, y = generate_data(sample_size, true_params, noise_std, rng)

        # 估计一次
        metrics = estimate_once(X, y)

        # 添加模拟次数标识（可选）
        metrics["sim_id"] = i

        results_list.append(metrics)

    # 转换为 DataFrame
    df_results = pd.DataFrame(results_list)

    return df_results


# ==========================================
# 4. Analysis (分析与报告物料生成)
# ==========================================
def analysis(results_df, true_beta1, output_file="src/week02/beta_dist.png"):
    """
    分析模拟结果，计算偏差、方差，输出 Markdown 表格和分布图

    参数:
        results_df (pd.DataFrame): 包含多次模拟结果的 DataFrame
        true_beta1 (float): 真实的 beta_1 值
        output_file (str): 图片保存路径（相对于当前工作目录）
    """
    # 设置 matplotlib 支持中文显示
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 计算偏差和方差
    beta1_est = results_df["beta_1_manual"]
    bias = np.mean(beta1_est) - true_beta1
    variance = np.var(beta1_est, ddof=1)

    print("\n### 手动估计量的偏差与方差 (基于 {} 次模拟)".format(len(results_df)))
    print("| 指标 | 数值 |")
    print("|------|------|")
    print(f"| 真实 β₁ | {true_beta1:.4f} |")
    print(f"| 估计期望 | {np.mean(beta1_est):.4f} |")
    print(f"| 偏差 | {bias:.4f} |")
    print(f"| 方差 | {variance:.6f} |")

    # 2. 方法对比表（取第一次模拟的结果展示）
    first = results_df.iloc[0]
    print("\n### 三种方法估计结果对比 (单次模拟示例)")
    print("| 方法 | β₀ | β₁ | R² (仅 sklearn) |")
    print("|------|----|----|-----------------|")
    print(f"| 手动公式 | {first['beta_0_manual']:.4f} | {first['beta_1_manual']:.4f} | - |")
    print(f"| sklearn | {first['beta_0_sklearn']:.4f} | {first['beta_1_sklearn']:.4f} | {first['r2_sklearn']:.4f} |")
    print(f"| statsmodels | {first['beta_0_statsmodels']:.4f} | {first['beta_1_statsmodels']:.4f} | - |")

    # 3. 假设检验结果汇总
    avg_pvalue = results_df["p_value_beta1"].mean()
    avg_fstat = results_df["f_statistic"].mean()
    avg_fpvalue = results_df["f_pvalue"].mean()
    print("\n### 假设检验与方差分析 (基于 {} 次模拟的平均结果)".format(len(results_df)))
    print("| 检验项 | 统计量 (平均) | p-value (平均) |")
    print("|--------|---------------|----------------|")
    print(f"| β₁ = 0 的 t 检验 | - | {avg_pvalue:.4e} |")
    print(f"| 整体 F 检验 | {avg_fstat:.2f} | {avg_fpvalue:.4e} |")

    # 4. 绘制分布图
    plt.figure(figsize=(8, 5))
    plt.hist(beta1_est, bins=30, density=True, alpha=0.7, edgecolor='black')
    plt.axvline(true_beta1, color='red', linestyle='--', label=f'真实 β₁ = {true_beta1}')
    plt.xlabel("β₁ 估计值")
    plt.ylabel("密度")
    plt.title("蒙特卡洛模拟: β₁ 估计量的分布")
    plt.legend()
    plt.grid(alpha=0.3)

    # 保存图片
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 自动创建目录
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"\n### 图像已保存至: {output_file}")