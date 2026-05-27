"""模块: utils.diagnostics
用途: 模型诊断工具箱 —— 多重共线性检测（VIF）。
"""
import numpy as np

from .models import AnalyticalOLS


def calculate_vif(X: np.ndarray) -> list:
    """计算矩阵 X 每一列特征的方差膨胀因子 (VIF)。

    原理: 对第 j 列特征，用其余所有特征对其进行 OLS 回归，得到 R²_j，
    然后 VIF_j = 1 / (1 - R²_j)。VIF 越大说明该特征与其他特征的
    线性关系越强，通常 VIF > 10 被认为存在严重多重共线性。

    参数:
        X: 形状为 (n_samples, n_features) 的特征矩阵（不应包含截距列）。

    返回:
        每个特征对应的 VIF 值列表，长度等于 X 的列数。
    """
    n_features = X.shape[1]
    vif_values = []

    # 逐列计算 VIF
    for j in range(n_features):
        # 取第 j 列作为"目标变量"
        y_j = X[:, j]

        # 剩余列作为"自变量"（即用其他特征来预测第 j 个特征）
        X_other = np.delete(X, j, axis=1)

        # 在自变量矩阵前添加截距列（全 1），否则回归不含常数项
        X_other_with_intercept = np.column_stack([np.ones(X_other.shape[0]), X_other])

        # 用 OLS 拟合，并计算该回归的 R²
        model = AnalyticalOLS().fit(X_other_with_intercept, y_j)
        r2_j = model.score(X_other_with_intercept, y_j)

        # VIF = 1 / (1 - R²)，若 R²=1 则 VIF 无穷大
        vif_j = 1.0 / (1.0 - r2_j) if r2_j < 1.0 else float("inf")
        vif_values.append(vif_j)

    return vif_values
