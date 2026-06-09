import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================
# 1. 路径设置
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "synthetic_highdim.csv")
SYNTHETIC_REPORT_PATH = os.path.join(RESULTS_DIR, "synthetic_report.md")
SUMMARY_REPORT_PATH = os.path.join(RESULTS_DIR, "summary_comparison.md")

FIG_ERROR_BY_P = os.path.join(FIGURES_DIR, "ols_error_by_p.png")
FIG_MATRIX_BY_P = os.path.join(FIGURES_DIR, "matrix_rank_condition.png")
FIG_COEF_STABILITY = os.path.join(FIGURES_DIR, "ols_coefficient_stability.png")
FIG_PCA_CUMVAR = os.path.join(FIGURES_DIR, "pca_cumulative_variance.png")
FIG_PCR_K = os.path.join(FIGURES_DIR, "pcr_rmse_by_k.png")
FIG_LASSO_PCR_COMPARE = os.path.join(FIGURES_DIR, "lasso_vs_pcr_comparison.png")


# ============================================================
# 2. 基础工具函数
# ============================================================

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def df_to_markdown(df, float_digits=4):
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x: f"{x:.{float_digits}f}")

    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")

    return "\n".join(lines)


def safe_condition_number(X):
    """
    自定义病态程度指标：条件数 condition number。
    条件数越大，说明矩阵越接近病态，OLS 系数越可能不稳定。
    """
    X = np.asarray(X, dtype=float)
    singular_values = np.linalg.svd(X, compute_uv=False)

    max_s = np.max(singular_values)
    min_s = np.min(singular_values)

    if min_s < 1e-12:
        return np.inf

    return max_s / min_s


def make_ols_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])


def make_pcr_pipeline(k):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=k)),
        ("model", LinearRegression())
    ])


# ============================================================
# 3. Task A：生成高维低秩模拟数据
# ============================================================

def make_latent_factor_data(n_samples=160, n_features=100, n_factors=5, noise_x=0.15, noise_y=1.0, random_state=42):
    """
    生成高维 + 潜在低秩结构数据。

    生成机制：
    1. 先生成少数 latent factors，记为 Z；
    2. 用 Z 线性组合生成大量原始特征 X；
    3. y 主要由少数 latent factors 决定。
    """
    rng = np.random.default_rng(random_state)

    Z = rng.normal(0, 1, size=(n_samples, n_factors))
    loadings = rng.normal(0, 1, size=(n_factors, n_features))

    X = Z @ loadings + rng.normal(0, noise_x, size=(n_samples, n_features))

    beta_z = np.array([5.0, -3.5, 2.0, 0.0, 1.5])
    beta_z = beta_z[:n_factors]

    y = Z @ beta_z + rng.normal(0, noise_y, size=n_samples)

    columns = [f"x{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["y"] = y

    return df


# ============================================================
# 4. A3：不同 p 下 OLS 训练误差和测试误差
# ============================================================

def run_ols_highdim_experiment():
    records = []
    p_list = [10, 30, 60, 120]

    for p in p_list:
        df = make_latent_factor_data(
            n_samples=160,
            n_features=p,
            n_factors=5,
            random_state=100 + p
        )

        X = df.drop(columns=["y"]).values
        y = df["y"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=2026
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        rank = np.linalg.matrix_rank(X_train_scaled)
        condition_number = safe_condition_number(X_train_scaled)

        records.append({
            "特征维度p": p,
            "训练样本数n_train": X_train.shape[0],
            "rank(X_train)": rank,
            "condition_number": condition_number,
            "train_RMSE": rmse(y_train, train_pred),
            "test_RMSE": rmse(y_test, test_pred),
        })

    result_df = pd.DataFrame(records)

    plt.figure(figsize=(8, 5))
    plt.plot(result_df["特征维度p"], result_df["train_RMSE"], marker="o", label="Train RMSE")
    plt.plot(result_df["特征维度p"], result_df["test_RMSE"], marker="o", label="Test RMSE")
    plt.xlabel("Number of features p")
    plt.ylabel("RMSE")
    plt.title("OLS Train/Test RMSE as Feature Dimension Increases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_ERROR_BY_P, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(result_df["特征维度p"], result_df["rank(X_train)"], marker="o", label="rank(X_train)")
    plt.xlabel("Number of features p")
    plt.ylabel("Matrix rank")
    plt.title("Rank of Training Design Matrix")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_MATRIX_BY_P, dpi=200)
    plt.close()

    return result_df


# ============================================================
# 5. A4：重复切分展示 OLS 系数不稳定
# ============================================================

def run_coefficient_stability_experiment(df, n_repeats=50):
    X = df.drop(columns=["y"]).values
    y = df["y"].values

    key_indices = [0, 1, 2, 3, 4]
    key_names = [f"x{i+1}" for i in key_indices]

    records = []

    for seed in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=seed
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        pred = model.predict(X_test_scaled)
        test_error = rmse(y_test, pred)

        for idx, name in zip(key_indices, key_names):
            records.append({
                "seed": seed,
                "变量": name,
                "系数": model.coef_[idx],
                "test_RMSE": test_error
            })

    coef_df = pd.DataFrame(records)

    data = []
    labels = []

    for name in key_names:
        data.append(coef_df[coef_df["变量"] == name]["系数"].values)
        labels.append(name)

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xlabel("Variable")
    plt.ylabel("OLS coefficient across random splits")
    plt.title("OLS Coefficient Instability across 50 Random Splits")
    plt.tight_layout()
    plt.savefig(FIG_COEF_STABILITY, dpi=200)
    plt.close()

    stability_summary = (
        coef_df
        .groupby("变量")
        .agg(
            系数均值=("系数", "mean"),
            系数标准差=("系数", "std"),
            test_RMSE均值=("test_RMSE", "mean"),
            test_RMSE标准差=("test_RMSE", "std")
        )
        .reset_index()
    )

    return coef_df, stability_summary


# ============================================================
# 6. Task B：PCA 和 PCR
# ============================================================

def run_pca_analysis(df):
    X = df.drop(columns=["y"]).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    pca_df = pd.DataFrame({
        "主成分个数k": np.arange(1, len(cumulative_variance) + 1),
        "累计解释方差比例": cumulative_variance
    })

    plt.figure(figsize=(8, 5))
    plt.plot(pca_df["主成分个数k"], pca_df["累计解释方差比例"], marker="o")
    plt.axhline(0.80, linestyle="--", label="80% variance")
    plt.axhline(0.90, linestyle="--", label="90% variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance ratio")
    plt.title("PCA Cumulative Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PCA_CUMVAR, dpi=200)
    plt.close()

    k_80 = int(np.argmax(cumulative_variance >= 0.80) + 1)
    k_90 = int(np.argmax(cumulative_variance >= 0.90) + 1)

    return pca_df, k_80, k_90


def cross_val_rmse_for_pcr(X, y, k, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = make_pcr_pipeline(k)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        scores.append(rmse(y_val, pred))

    return float(np.mean(scores))


def run_pcr_experiment(df, max_k=20):
    X = df.drop(columns=["y"]).values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=2026
    )

    records = []

    for k in range(1, max_k + 1):
        model = make_pcr_pipeline(k)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        cv_score = cross_val_rmse_for_pcr(X_train, y_train, k)

        records.append({
            "主成分个数k": k,
            "PCR_train_RMSE": rmse(y_train, train_pred),
            "PCR_test_RMSE": rmse(y_test, test_pred),
            "PCR_CV_RMSE": cv_score
        })

    pcr_df = pd.DataFrame(records)

    best_row = pcr_df.sort_values("PCR_CV_RMSE").iloc[0]
    best_k = int(best_row["主成分个数k"])

    plt.figure(figsize=(9, 5))
    plt.plot(pcr_df["主成分个数k"], pcr_df["PCR_train_RMSE"], marker="o", label="PCR Train RMSE")
    plt.plot(pcr_df["主成分个数k"], pcr_df["PCR_test_RMSE"], marker="o", label="PCR Test RMSE")
    plt.plot(pcr_df["主成分个数k"], pcr_df["PCR_CV_RMSE"], marker="o", label="PCR CV RMSE")
    plt.axvline(best_k, linestyle="--", label=f"Best k by CV = {best_k}")
    plt.xlabel("Number of principal components k")
    plt.ylabel("RMSE")
    plt.title("PCR RMSE under Different Numbers of Principal Components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PCR_K, dpi=200)
    plt.close()

    return pcr_df, best_k


# ============================================================
# 7. Task C：Lasso vs PCR
# ============================================================

def make_sparse_truth_data(n_samples=180, n_features=100, random_state=123):
    rng = np.random.default_rng(random_state)

    X = rng.normal(0, 1, size=(n_samples, n_features))
    beta = np.zeros(n_features)
    beta[[0, 5, 12, 30, 45]] = [5.0, -4.0, 3.0, 2.5, -2.0]

    y = X @ beta + rng.normal(0, 1.0, size=n_samples)

    columns = [f"x{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["y"] = y

    return df


def evaluate_lasso_and_pcr(df, scenario_name):
    X = df.drop(columns=["y"]).values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=2026
    )

    lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LassoCV(cv=5, random_state=42, max_iter=20000))
    ])

    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    lasso_coef = lasso.named_steps["model"].coef_
    lasso_nonzero = int(np.sum(np.abs(lasso_coef) > 1e-6))

    # PCR 选择 k
    candidate_ks = list(range(1, 21))
    cv_records = []

    for k in candidate_ks:
        cv_rmse = cross_val_rmse_for_pcr(X_train, y_train, k)
        cv_records.append((k, cv_rmse))

    best_k = sorted(cv_records, key=lambda x: x[1])[0][0]

    pcr = make_pcr_pipeline(best_k)
    pcr.fit(X_train, y_train)
    pcr_pred = pcr.predict(X_test)

    records = [
        {
            "场景": scenario_name,
            "方法": "Lasso",
            "test_RMSE": rmse(y_test, lasso_pred),
            "test_MAE": mean_absolute_error(y_test, lasso_pred),
            "复杂度指标": "非零系数个数",
            "复杂度数值": lasso_nonzero,
        },
        {
            "场景": scenario_name,
            "方法": "PCR",
            "test_RMSE": rmse(y_test, pcr_pred),
            "test_MAE": mean_absolute_error(y_test, pcr_pred),
            "复杂度指标": "保留主成分个数k",
            "复杂度数值": best_k,
        }
    ]

    # 稳定性：重复切分下 test RMSE 的标准差
    stability_records = []

    for seed in range(30):
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X, y, test_size=0.30, random_state=seed
        )

        lasso_s = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LassoCV(cv=5, random_state=42, max_iter=20000))
        ])
        lasso_s.fit(X_train_s, y_train_s)
        lasso_s_pred = lasso_s.predict(X_test_s)

        pcr_s = make_pcr_pipeline(best_k)
        pcr_s.fit(X_train_s, y_train_s)
        pcr_s_pred = pcr_s.predict(X_test_s)

        stability_records.append({
            "场景": scenario_name,
            "方法": "Lasso",
            "重复编号": seed,
            "test_RMSE": rmse(y_test_s, lasso_s_pred)
        })

        stability_records.append({
            "场景": scenario_name,
            "方法": "PCR",
            "重复编号": seed,
            "test_RMSE": rmse(y_test_s, pcr_s_pred)
        })

    stability_df = pd.DataFrame(stability_records)

    stability_summary = (
        stability_df
        .groupby(["场景", "方法"])["test_RMSE"]
        .std()
        .reset_index()
        .rename(columns={"test_RMSE": "稳定性指标_test_RMSE标准差"})
    )

    result_df = pd.DataFrame(records)
    result_df = result_df.merge(stability_summary, on=["场景", "方法"], how="left")

    return result_df


def run_lasso_vs_pcr_comparison():
    sparse_df = make_sparse_truth_data()
    latent_df = make_latent_factor_data(n_samples=180, n_features=100, n_factors=5, random_state=456)

    sparse_result = evaluate_lasso_and_pcr(sparse_df, "Sparse truth")
    latent_result = evaluate_lasso_and_pcr(latent_df, "Latent-factor truth")

    comparison_df = pd.concat([sparse_result, latent_result], ignore_index=True)

    plot_df = comparison_df.copy()

    scenarios = plot_df["场景"].unique()
    methods = ["Lasso", "PCR"]

    x_positions = np.arange(len(scenarios))
    width = 0.35

    plt.figure(figsize=(8, 5))

    for i, method in enumerate(methods):
        values = []
        for scenario in scenarios:
            row = plot_df[(plot_df["场景"] == scenario) & (plot_df["方法"] == method)]
            values.append(float(row["test_RMSE"].iloc[0]))

        plt.bar(x_positions + (i - 0.5) * width, values, width=width, label=method)

    plt.xticks(x_positions, scenarios)
    plt.ylabel("Test RMSE")
    plt.title("Lasso vs PCR under Two Data-Generating Worlds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_LASSO_PCR_COMPARE, dpi=200)
    plt.close()

    return comparison_df


# ============================================================
# 8. 写报告
# ============================================================

def write_synthetic_report(
    df,
    ols_dim_df,
    coef_stability_summary,
    pca_df,
    k_80,
    k_90,
    pcr_df,
    best_k
):
    lines = []

    lines.append("# Week 14 Synthetic Report：高维回归、PCA 与 PCR")
    lines.append("")
    lines.append("## 1. 数据生成机制")
    lines.append("")
    lines.append("本次作业生成了一份高维且带有潜在低秩结构的模拟数据。")
    lines.append("")
    lines.append(f"- 样本量：{df.shape[0]}")
    lines.append(f"- 特征数：{df.shape[1] - 1}")
    lines.append("- 潜在因子数量：5")
    lines.append("- 目标变量 y 主要由少数潜在因子驱动，而不是由所有原始变量独立决定。")
    lines.append("")
    lines.append("这份数据可以称为“高维 + 信息冗余”数据，因为虽然原始变量很多，但是它们主要来自少数几个 latent factors 的线性组合。换句话说，很多原始变量其实在重复表达相似的信息。")
    lines.append("")
    lines.append("数据文件保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week14/data/synthetic_highdim.csv")
    lines.append("```")
    lines.append("")
    lines.append("## 2. OLS 在高维场景下的问题")
    lines.append("")
    lines.append("不同特征维度下的 OLS 结果如下：")
    lines.append("")
    lines.append(df_to_markdown(ols_dim_df, float_digits=4))
    lines.append("")
    lines.append("对应图像：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week14/figures/ols_error_by_p.png")
    lines.append("src/week14/figures/matrix_rank_condition.png")
    lines.append("```")
    lines.append("")
    lines.append("第一张图的横轴是特征维度 p，纵轴是 RMSE，图中同时展示了训练集 RMSE 和测试集 RMSE。第二张图的横轴是特征维度 p，纵轴是训练矩阵的 rank。")
    lines.append("")
    lines.append("当 p 增大，特别是 p 接近或超过训练样本量时，OLS 可能会把训练集拟合得非常好，甚至出现接近 0 的训练误差。但这并不说明模型真的好，因为测试误差可能仍然很大。这种“训练误差很低”反而是危险信号，说明模型可能只是记住了训练数据里的噪声。")
    lines.append("")
    lines.append("## 3. OLS 系数稳定性")
    lines.append("")
    lines.append("对固定数据集重复 50 次随机切分后，部分变量的系数波动如下：")
    lines.append("")
    lines.append(df_to_markdown(coef_stability_summary, float_digits=4))
    lines.append("")
    lines.append("系数稳定性图保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week14/figures/ols_coefficient_stability.png")
    lines.append("```")
    lines.append("")
    lines.append("这张图的横轴是不同变量，纵轴是 OLS 在不同随机切分下得到的系数值。箱线越高、范围越大，说明这个变量的系数越不稳定。")
    lines.append("")
    lines.append("我观察到，不只是测试误差会波动，系数本身也会明显波动。系数不稳定是一种重要风险，因为它说明模型对样本划分非常敏感。如果换一批样本，变量的重要性解释就可能发生变化。")
    lines.append("")
    lines.append("## 4. PCA 累计解释方差")
    lines.append("")
    lines.append(f"PCA 结果显示，前 {k_80} 个主成分可以解释至少 80% 的方差，前 {k_90} 个主成分可以解释至少 90% 的方差。")
    lines.append("")
    lines.append("累计解释方差图保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week14/figures/pca_cumulative_variance.png")
    lines.append("```")
    lines.append("")
    lines.append("这张图的横轴是主成分个数，纵轴是累计解释方差比例。图中可以看出，少数几个主成分已经解释了大部分原始变量的信息，因此原始高维空间其实更接近一个低维子空间。")
    lines.append("")
    lines.append("## 5. PCR 实验")
    lines.append("")
    lines.append("PCR 在不同主成分个数 k 下的表现如下：")
    lines.append("")
    lines.append(df_to_markdown(pcr_df, float_digits=4))
    lines.append("")
    lines.append(f"根据交叉验证 RMSE，本次选择的最佳主成分个数为 k = {best_k}。")
    lines.append("")
    lines.append("PCR 曲线图保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week14/figures/pcr_rmse_by_k.png")
    lines.append("```")
    lines.append("")
    lines.append("这张图的横轴是保留的主成分个数 k，纵轴是 RMSE，图中展示了 train RMSE、test RMSE 和 CV RMSE。PCR CV RMSE 表示在训练集内部通过交叉验证估计出来的预测误差，它可以帮助我们选择更合适的 k。")
    lines.append("")
    lines.append("## 6. 公式与定义")
    lines.append("")
    lines.append("OLS 的估计式为：")
    lines.append("")
    lines.append("```text")
    lines.append("beta_hat = (X^T X)^(-1) X^T y")
    lines.append("```")
    lines.append("")
    lines.append("第一主成分可以理解为寻找一个方向 v，使得投影后的方差最大：")
    lines.append("")
    lines.append("```text")
    lines.append("v1 = argmax Var(Xv),  subject to ||v|| = 1")
    lines.append("```")
    lines.append("")
    lines.append("PCR 的流程可以写成：")
    lines.append("")
    lines.append("```text")
    lines.append("Z_k = X V_k")
    lines.append("y = Z_k gamma + error")
    lines.append("```")
    lines.append("")
    lines.append("其中 V_k 是前 k 个主成分方向，Z_k 是原始 X 投影到主成分空间后的低维表示。PCR 的核心思想不是直接在原始高维变量上回归，而是先压缩信息，再回归。")

    with open(SYNTHETIC_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_summary_report(comparison_df):
    lines = []

    lines.append("# Week 14 Summary Comparison：Lasso vs PCR")
    lines.append("")
    lines.append("## 1. 两种数据世界")
    lines.append("")
    lines.append("本次比较了两种不同的数据生成机制：")
    lines.append("")
    lines.append("1. Sparse truth：只有少数原始变量真正直接决定 y，其他变量主要是噪声。")
    lines.append("2. Latent-factor truth：大量原始变量由少数潜在因子生成，y 也主要由这些潜在因子驱动。")
    lines.append("")
    lines.append("## 2. Lasso 与 PCR 结果对比")
    lines.append("")
    lines.append(df_to_markdown(comparison_df, float_digits=4))
    lines.append("")
    lines.append("对比图保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week14/figures/lasso_vs_pcr_comparison.png")
    lines.append("```")
    lines.append("")
    lines.append("这张图把两个场景分开展示，横轴是数据场景，纵轴是测试集 RMSE，每组柱子分别代表 Lasso 和 PCR。")
    lines.append("")
    lines.append("## 3. 核心问题回答")
    lines.append("")
    lines.append("### 3.1 当数据真的是 sparse truth 时，为什么 Lasso 更自然？")
    lines.append("")
    lines.append("因为 sparse truth 的特点是只有少数原始变量真正有用。Lasso 通过 L1 正则化可以把很多无关变量的系数压缩为 0，所以它天然适合做变量筛选。")
    lines.append("")
    lines.append("### 3.2 当数据更像 latent-factor truth 时，为什么 PCR 更自然？")
    lines.append("")
    lines.append("因为 latent-factor truth 里，很多原始变量都在重复表达少数潜在因子的信息。这时问题的重点不是从原始变量中挑出某几列，而是把这些重复信息压缩成少数几个主成分。PCR 正好适合这种信息压缩任务。")
    lines.append("")
    lines.append("### 3.3 Lasso 回答“谁留下”，PCR 回答什么？")
    lines.append("")
    lines.append("Lasso 更像是在回答：哪些原始变量应该留下。PCR 更像是在回答：能不能把原始高维信息压缩成几个更稳定的综合方向。")
    lines.append("")
    lines.append("### 3.4 如果业务方要求一个更短的变量名单，用哪个方法？")
    lines.append("")
    lines.append("更可能使用 Lasso。因为 Lasso 可以直接给出非零系数变量，也就是一个更短的变量名单。")
    lines.append("")
    lines.append("### 3.5 如果业务方要求一个更稳的预测器，用哪个方法？")
    lines.append("")
    lines.append("如果数据存在明显共线性或潜在因子结构，我更可能使用 PCR。因为 PCR 通过主成分压缩可以减少冗余信息带来的不稳定。")
    lines.append("")
    lines.append("### 3.6 为什么本周主线更适合比较 Lasso vs PCR？")
    lines.append("")
    lines.append("因为本周的核心是 selection vs compression。Lasso 是典型的变量筛选方法，PCR 是典型的信息压缩方法。前向/后向选择也属于 selection 路线，但它们不能很好地体现 PCA/PCR 这种主成分压缩思想。")
    lines.append("")
    lines.append("如果一定要加入前向/后向选择，它们更接近 selection 路线，而不是 compression 路线。")

    with open(SUMMARY_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 9. 主程序
# ============================================================

def main():
    print("=" * 70)
    print("Week 14：高维回归、PCA 与 PCR")
    print("=" * 70)

    print("\n[阶段1] 生成高维低秩模拟数据...")
    df = make_latent_factor_data(n_samples=160, n_features=100, n_factors=5, random_state=42)
    df.to_csv(DATA_PATH, index=False)
    print(f"数据已保存：{DATA_PATH}")
    print(f"数据规模：{df.shape}")

    print("\n[阶段2] OLS 高维实验...")
    ols_dim_df = run_ols_highdim_experiment()
    print(ols_dim_df)

    print("\n[阶段3] OLS 系数稳定性实验...")
    coef_df, coef_stability_summary = run_coefficient_stability_experiment(df, n_repeats=50)
    print(coef_stability_summary)

    print("\n[阶段4] PCA 累计解释方差分析...")
    pca_df, k_80, k_90 = run_pca_analysis(df)
    print(f"解释 80% 方差所需主成分个数：{k_80}")
    print(f"解释 90% 方差所需主成分个数：{k_90}")

    print("\n[阶段5] PCR 实验...")
    pcr_df, best_k = run_pcr_experiment(df, max_k=20)
    print(pcr_df)
    print(f"PCR 最优 k：{best_k}")

    print("\n[阶段6] Lasso vs PCR 对比...")
    comparison_df = run_lasso_vs_pcr_comparison()
    print(comparison_df)

    print("\n[阶段7] 写报告...")
    write_synthetic_report(
        df=df,
        ols_dim_df=ols_dim_df,
        coef_stability_summary=coef_stability_summary,
        pca_df=pca_df,
        k_80=k_80,
        k_90=k_90,
        pcr_df=pcr_df,
        best_k=best_k
    )

    write_summary_report(comparison_df)

    print("\n全部任务完成！")
    print("生成文件如下：")
    print(f"1. {DATA_PATH}")
    print(f"2. {SYNTHETIC_REPORT_PATH}")
    print(f"3. {SUMMARY_REPORT_PATH}")
    print(f"4. {FIG_ERROR_BY_P}")
    print(f"5. {FIG_MATRIX_BY_P}")
    print(f"6. {FIG_COEF_STABILITY}")
    print(f"7. {FIG_PCA_CUMVAR}")
    print(f"8. {FIG_PCR_K}")
    print(f"9. {FIG_LASSO_PCR_COMPARE}")


if __name__ == "__main__":
    main()