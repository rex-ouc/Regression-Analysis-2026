#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
MPL_DIR = Path(tempfile.gettempdir()) / "mxt_week15_mplconfig"
for path in (DATA_DIR, RESULTS_DIR, FIG_DIR, MPL_DIR):
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(SRC_DIR))
from utils.metrics import binary_log_loss, classification_metrics_at_threshold


RANDOM_SEED = 42
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def configure_matplotlib_fonts() -> None:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/mnt/c/Windows/Fonts/simhei.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120


configure_matplotlib_fonts()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def rounded(value, digits: int = 4):
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    return value


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: f"{x:.{digits}f}")
    columns = [str(col) for col in formatted.columns]
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in formatted.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in formatted.columns) + " |")
    return "\n".join(rows)


def make_synthetic_binary(n_samples: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    x1 = rng.normal(size=n_samples)
    x2 = rng.normal(size=n_samples)
    x3 = rng.normal(size=n_samples)
    x4 = rng.normal(size=n_samples)
    x5 = 0.75 * x1 + rng.normal(scale=0.45, size=n_samples)
    x6 = rng.normal(size=n_samples)
    eta = -0.35 + 1.45 * x1 - 1.15 * x2 + 0.65 * x3 + 0.25 * x1 * x3
    probability = sigmoid(eta)
    y = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "x1_positive_driver": x1,
            "x2_negative_driver": x2,
            "x3_positive_driver": x3,
            "x4_noise": x4,
            "x5_correlated_with_x1": x5,
            "x6_noise": x6,
            "true_probability": probability,
            "y": y,
        }
    )


def evaluate_model(y_true, probability, threshold: float = 0.5) -> dict[str, float]:
    metrics = classification_metrics_at_threshold(y_true, probability, threshold)
    return {
        **metrics,
        "ROC_AUC": float(roc_auc_score(y_true, probability)),
        "log_loss": float(binary_log_loss(y_true, probability)),
    }


def threshold_scan(y_true, probability) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.1, 1.0, 0.1), 2):
        rows.append({"threshold": float(threshold), **classification_metrics_at_threshold(y_true, probability, threshold)})
    return pd.DataFrame(rows)


def run_synthetic_tasks(synthetic_df: pd.DataFrame):
    features = [
        "x1_positive_driver",
        "x2_negative_driver",
        "x3_positive_driver",
        "x4_noise",
        "x5_correlated_with_x1",
        "x6_noise",
    ]
    X = synthetic_df[features]
    y = synthetic_df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=RANDOM_SEED, stratify=y
    )

    linear_model = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    logistic_model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))])
    linear_model.fit(X_train, y_train)
    logistic_model.fit(X_train, y_train)

    linear_output = linear_model.predict(X_test)
    logistic_probability = logistic_model.predict_proba(X_test)[:, 1]
    linear_metrics = classification_metrics_at_threshold(y_test, linear_output, 0.5)
    linear_metrics.update(
        {
            "outside_0_1_rate": float(np.mean((linear_output < 0) | (linear_output > 1))),
            "min_output": float(linear_output.min()),
            "max_output": float(linear_output.max()),
        }
    )
    logistic_metrics = evaluate_model(y_test, logistic_probability, 0.5)
    threshold_df = threshold_scan(y_test, logistic_probability)

    grid = np.linspace(X["x1_positive_driver"].quantile(0.01), X["x1_positive_driver"].quantile(0.99), 220)
    base = pd.DataFrame(np.repeat([X.mean().to_numpy()], len(grid), axis=0), columns=features)
    base["x1_positive_driver"] = grid
    linear_grid = linear_model.predict(base)
    logistic_grid = logistic_model.predict_proba(base)[:, 1]

    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(len(synthetic_df), size=160, replace=False)
    jitter = rng.normal(0, 0.025, len(sample_idx))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        synthetic_df.iloc[sample_idx]["x1_positive_driver"],
        synthetic_df.iloc[sample_idx]["y"] + jitter,
        s=18,
        alpha=0.45,
        color="tab:gray",
        label="真实 0/1 标签",
    )
    ax.plot(grid, linear_grid, color="tab:orange", linewidth=2, label="LinearRegression 输出")
    ax.plot(grid, logistic_grid, color="tab:blue", linewidth=2.2, label="LogisticRegression 概率")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
    ax.axhline(1, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlabel("x1_positive_driver")
    ax.set_ylabel("模型输出 / 正类概率")
    ax.set_title("线性回归输出与逻辑回归概率输出对比")
    ax.legend()
    fig.savefig(FIG_DIR / "linear_vs_logistic_output.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    p_grid = np.linspace(0.001, 0.999, 450)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    axes[0].plot(p_grid, (1 - p_grid) ** 2, label="平方误差 MSE", color="tab:orange")
    axes[0].plot(p_grid, -np.log(p_grid), label="log loss", color="tab:blue")
    axes[0].set_title("真实标签 y=1")
    axes[1].plot(p_grid, p_grid**2, label="平方误差 MSE", color="tab:orange")
    axes[1].plot(p_grid, -np.log(1 - p_grid), label="log loss", color="tab:blue")
    axes[1].set_title("真实标签 y=0")
    for ax in axes:
        ax.set_xlabel("预测为正类的概率 p")
        ax.set_ylabel("损失值")
        ax.set_ylim(0, 7)
        ax.legend()
    fig.suptitle("预测概率变化时 MSE 与 log loss 的惩罚差异")
    fig.savefig(FIG_DIR / "loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col in ["accuracy", "precision", "recall", "F1"]:
        ax.plot(threshold_df["threshold"], threshold_df[col], marker="o", label=col)
    ax.set_xlabel("分类阈值 threshold")
    ax.set_ylabel("指标值")
    ax.set_ylim(0, 1.05)
    ax.set_title("阈值变化下的分类指标权衡")
    ax.legend()
    fig.savefig(FIG_DIR / "threshold_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "linear_metrics": linear_metrics,
        "logistic_metrics": logistic_metrics,
        "threshold_df": threshold_df,
        "test_positive_rate": float(y_test.mean()),
    }


def make_regularized_data(n_samples: int = 520):
    rng = np.random.default_rng(2026)
    z1 = rng.normal(size=n_samples)
    z2 = rng.normal(size=n_samples)
    z3 = rng.normal(size=n_samples)
    data = {}
    for i in range(1, 9):
        data[f"group1_x{i:02d}"] = z1 + rng.normal(0, 0.25, n_samples)
    for i in range(1, 9):
        data[f"group2_x{i:02d}"] = z2 + rng.normal(0, 0.25, n_samples)
    for i in range(1, 5):
        data[f"group3_x{i:02d}"] = z3 + rng.normal(0, 0.35, n_samples)
    for i in range(1, 16):
        data[f"noise_x{i:02d}"] = rng.normal(size=n_samples)
    X = pd.DataFrame(data)
    eta = -0.2 + 1.3 * z1 - 1.0 * z2 + 0.8 * z3 + 0.5 * X["noise_x03"]
    probability = sigmoid(eta)
    y = pd.Series(rng.binomial(1, probability), name="y")
    return X, y, probability


def validation_select_regularized_model(X_train, y_train, penalty: str):
    X_inner, X_val, y_inner, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train
    )
    best_model = None
    best_c = None
    best_loss = np.inf
    for c_value in np.logspace(-2, 2, 9):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty=penalty,
                        C=float(c_value),
                        solver="liblinear",
                        max_iter=3000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        model.fit(X_inner, y_inner)
        val_probability = model.predict_proba(X_val)[:, 1]
        val_loss = binary_log_loss(y_val, val_probability)
        if val_loss < best_loss:
            best_model = model
            best_c = float(c_value)
            best_loss = val_loss
    best_model.fit(X_train, y_train)
    return best_model, best_c


def run_regularization_task():
    X, y, probability = make_regularized_data()
    high_dim_df = X.copy()
    high_dim_df["true_probability"] = probability
    high_dim_df["y"] = y
    high_dim_df.to_csv(DATA_DIR / "regularized_binary.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=RANDOM_SEED, stratify=y
    )
    rows = []
    coef_rows = []
    for penalty in ["l1", "l2"]:
        model, best_c = validation_select_regularized_model(X_train, y_train, penalty)
        probability_test = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, probability_test, 0.5)
        coefs = model.named_steps["model"].coef_.ravel()
        nonzero = int(np.sum(np.abs(coefs) > 1e-8))
        rows.append(
            {
                "model": f"{penalty.upper()} LogisticRegression",
                "penalty": penalty.upper(),
                "best_C": best_c,
                "accuracy": metrics["accuracy"],
                "recall": metrics["recall"],
                "ROC_AUC": metrics["ROC_AUC"],
                "log_loss": metrics["log_loss"],
                "nonzero_coefficients": nonzero,
            }
        )
        for feature, coef in zip(X.columns, coefs):
            coef_rows.append({"penalty": penalty.upper(), "feature": feature, "coef": coef})
    comparison = pd.DataFrame(rows)
    coef_df = pd.DataFrame(coef_rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    metrics = ["accuracy", "recall", "ROC_AUC"]
    x_pos = np.arange(len(metrics))
    width = 0.34
    for offset, penalty in [(-width / 2, "L1"), (width / 2, "L2")]:
        values = comparison.loc[comparison["penalty"] == penalty, metrics].iloc[0].to_numpy(float)
        ax.bar(x_pos + offset, values, width=width, label=f"{penalty} 正则化")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("指标值")
    ax.set_title("L1 与 L2 逻辑回归测试集性能")
    ax.legend()
    fig.savefig(FIG_DIR / "regularization_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(comparison["penalty"], comparison["nonzero_coefficients"], color=["tab:blue", "tab:orange"])
    axes[0].set_xlabel("正则化类型")
    axes[0].set_ylabel("非零系数个数")
    axes[0].set_title("模型稀疏度对比")
    for penalty, color in [("L1", "tab:blue"), ("L2", "tab:orange")]:
        values = coef_df.loc[coef_df["penalty"] == penalty, "coef"]
        axes[1].hist(values, bins=16, alpha=0.65, label=f"{penalty} 系数", color=color)
    axes[1].set_xlabel("系数大小")
    axes[1].set_ylabel("特征数量")
    axes[1].set_title("系数分布对比")
    axes[1].legend()
    fig.savefig(FIG_DIR / "regularization_complexity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return comparison, coef_df


def run_real_data_task():
    cancer = load_breast_cancer(as_frame=True)
    X = cancer.data.copy()
    y = pd.Series(1 - cancer.target.to_numpy(), name="malignant")
    real_df = X.copy()
    real_df["malignant"] = y
    real_df.to_csv(DATA_DIR / "real_binary_breast_cancer.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=4000, random_state=RANDOM_SEED)),
        ]
    )
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    metrics_05 = evaluate_model(y_test, probability, 0.5)
    scan_df = threshold_scan(y_test, probability)
    best_f1 = scan_df.loc[scan_df["F1"].idxmax()].to_dict()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col in ["accuracy", "precision", "recall", "F1"]:
        ax.plot(scan_df["threshold"], scan_df[col], marker="o", label=col)
    ax.set_xlabel("分类阈值 threshold")
    ax.set_ylabel("指标值")
    ax.set_ylim(0, 1.05)
    ax.set_title("真实乳腺癌数据中的阈值权衡")
    ax.legend()
    fig.savefig(FIG_DIR / "real_threshold_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "positive_rate": float(y.mean()),
        "metrics_05": metrics_05,
        "threshold_df": scan_df,
        "best_f1": best_f1,
    }


def write_reports(synthetic_df, synthetic_results, reg_comparison, real_results):
    linear = synthetic_results["linear_metrics"]
    logistic = synthetic_results["logistic_metrics"]
    threshold_df = synthetic_results["threshold_df"]
    reg_display = reg_comparison.rename(
        columns={
            "model": "模型",
            "penalty": "正则化",
            "best_C": "最优C",
            "accuracy": "accuracy",
            "recall": "recall",
            "ROC_AUC": "ROC-AUC",
            "log_loss": "log loss",
            "nonzero_coefficients": "非零系数个数",
        }
    )

    basic_metrics = pd.DataFrame(
        [
            {
                "模型": "LinearRegression 阈值0.5",
                "TP": linear["TP"],
                "TN": linear["TN"],
                "FP": linear["FP"],
                "FN": linear["FN"],
                "accuracy": linear["accuracy"],
                "precision": linear["precision"],
                "recall": linear["recall"],
                "F1": linear["F1"],
            },
            {
                "模型": "LogisticRegression 阈值0.5",
                "TP": logistic["TP"],
                "TN": logistic["TN"],
                "FP": logistic["FP"],
                "FN": logistic["FN"],
                "accuracy": logistic["accuracy"],
                "precision": logistic["precision"],
                "recall": logistic["recall"],
                "F1": logistic["F1"],
            },
        ]
    )
    threshold_display = threshold_df.rename(
        columns={
            "threshold": "阈值",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "F1": "F1",
        }
    )[["阈值", "accuracy", "precision", "recall", "F1"]]

    synthetic_report = f"""# Week 15 任务A：模拟二分类与逻辑回归

## A1-A2. 数据生成与 DGP

本次我生成了 `{len(synthetic_df)}` 个样本、6 个原始特征的模拟二分类数据，并保存到 `src/week15/data/synthetic_binary.csv`。其中 `x1_positive_driver`、`x3_positive_driver` 会提高正类概率，`x2_negative_driver` 会降低正类概率；`x4_noise` 和 `x6_noise` 主要是噪声特征，`x5_correlated_with_x1` 与 `x1` 明显相关，用来保留一点现实数据中常见的相关结构。

真实数据生成机制是先构造线性预测子：

$$
\\eta=-0.35+1.45x_1-1.15x_2+0.65x_3+0.25x_1x_3
$$

再通过 sigmoid 函数得到概率：

$$
p=\\frac{{1}}{{1+e^{{-\\eta}}}}
$$

最后从 Bernoulli(p) 中抽样得到 0/1 标签 `y`。因此，`y` 不是人为硬切出来的标签，而是先有概率、再有随机结果。

本数据正类比例为 `{synthetic_df["y"].mean():.4f}`。

## A3. LinearRegression 与 LogisticRegression 对比

| 指标 | LinearRegression | LogisticRegression |
|:--|--:|--:|
| accuracy | {linear["accuracy"]:.4f} | {logistic["accuracy"]:.4f} |
| precision | {linear["precision"]:.4f} | {logistic["precision"]:.4f} |
| recall | {linear["recall"]:.4f} | {logistic["recall"]:.4f} |
| F1 | {linear["F1"]:.4f} | {logistic["F1"]:.4f} |

LinearRegression 在阈值 0.5 下也能勉强产生分类结果，但它的原始输出不是概率。本次测试集中，线性回归输出低于 0 或高于 1 的比例为 `{linear["outside_0_1_rate"]:.4f}`，最小输出为 `{linear["min_output"]:.4f}`，最大输出为 `{linear["max_output"]:.4f}`。这说明如果强行把它解释成“正类概率”，会出现概率越界的问题。

## A4. 核心对比图

图 `figures/linear_vs_logistic_output.png` 展示两类模型输出行为：

- 横轴是主要特征 `x1_positive_driver`。
- 纵轴是模型输出或正类概率。
- 灰色散点是真实 0/1 标签，加入了很小的纵向抖动，便于观察。
- 橙色线是 LinearRegression 的连续输出。
- 蓝色线是 LogisticRegression 的正类概率输出。

这张图支持的结论是：线性回归输出可以穿出 `[0,1]` 区间，天然不是概率；逻辑回归通过 sigmoid 把输出限制在 `[0,1]` 内，更适合解释为“发生概率”。

## A5. 核心问题回答

LinearRegression 在二分类任务里最不自然的地方，不是它完全不能分类，而是它的输出没有概率意义。它优化的是连续数值误差，输出可以小于 0 或大于 1，因此只能事后硬加阈值。

逻辑回归的输出更容易解释成概率，是因为它先计算线性预测子，再通过 sigmoid 映射到 `[0,1]`。这个输出可以直接理解为 `P(y=1|x)`。

所以这里的关键区别不是“能不能分类”，而是“模型输出是否有概率意义”。线性回归加阈值可以分出类别，但逻辑回归同时给出了概率、阈值和不确定性的解释空间。
"""

    threshold_report = f"""# Week 15 任务B-C：log loss、混淆矩阵与阈值权衡

## B1. Bernoulli、似然与 log loss

Bernoulli 分布：

$$
Y \\sim Bernoulli(p)
$$

这里的 `Y` 只能取 0 或 1，`p` 表示取 1 的概率。在二分类问题里，模型真正需要估计的不是一个任意连续数值，而是样本属于正类的概率。

单样本概率：

$$
L(p;y)=p^y(1-p)^{{1-y}}
$$

如果真实标签 `y=1`，这个式子变成 `p`；如果真实标签 `y=0`，它变成 `1-p`。也就是说，模型给真实类别的概率越高，似然就越大。

单样本负对数似然，也就是 log loss：

$$
-\\log L(p;y)=-\\left[y\\log(p)+(1-y)\\log(1-p)\\right]
$$

训练时最小化 log loss 等价于最大化 Bernoulli likelihood。它不是随便指定的损失函数，而是从“二分类标签服从 Bernoulli 分布”这个统计假设自然推出来的。

## B2-B3. 损失曲线解释

图 `figures/loss_curves.png` 比较了两种 loss：

- 横轴是模型预测为正类的概率 `p`。
- 纵轴是损失值。
- 左图固定真实标签 `y=1`，右图固定真实标签 `y=0`。
- 橙色线是平方误差 MSE，蓝色线是 log loss。

当模型“错得很自信”时，log loss 惩罚更重。例如真实 `y=1` 却给出很小的 `p`，或者真实 `y=0` 却给出接近 1 的 `p`，log loss 会迅速变大。这很合理，因为概率模型如果非常自信地给错方向，应当承担更高代价。

如果我们已经把输出解释成概率，那么 log loss 比 MSE 更自然。MSE 把 0/1 当成普通连续数值来拟合，而 log loss 直接对应 Bernoulli likelihood，更贴合分类建模本身。

## C1. 混淆矩阵和基础指标

下面表格基于测试集，阈值固定为 0.5：

{markdown_table(basic_metrics)}

## C2-C3. threshold 扫描

下面表格展示不同阈值下 LogisticRegression 的分类指标：

{markdown_table(threshold_display)}

图 `figures/threshold_metrics.png` 展示阈值曲线：

- 横轴是分类阈值 threshold。
- 纵轴是指标值。
- accuracy、precision、recall、F1 四条线分别表示对应指标。

随着阈值升高，模型会更谨慎地预测正类。通常 precision 会变高，因为被判为正类的样本更“确定”；recall 往往会下降，因为更多真实正类会被漏掉。accuracy 和 F1 会在中间区域出现折中点。

## C4. 业务场景：疾病初筛

我选择“疾病初筛”场景。在这个场景里，我最在意 recall，因为漏掉真正患病的人代价很高。precision 也重要，但初筛阶段通常宁愿多召回一些疑似样本，再交给后续检查确认。

如果向业务方推荐阈值，我会优先选择 recall 较高且 F1 不太差的阈值。从本次阈值扫描看，阈值 0.3 左右能明显提高召回，同时 F1 仍保持较好水平。因此我会解释为：初筛模型不要太保守，先把高风险人群尽量捞出来，再用后续流程降低误报成本。
"""

    regularization_report = f"""# Week 15 任务D：正则化逻辑回归 L1 vs L2

## D1. 高维与共线性数据

我构造了一份新的二分类数据 `src/week15/data/regularized_binary.csv`。它有 520 个样本、35 个特征，满足“特征数不少于 20”的要求。数据中包含三组相关特征：`group1_*` 共享潜在因子 `z1`，`group2_*` 共享潜在因子 `z2`，`group3_*` 共享潜在因子 `z3`，同时加入 15 个噪声特征。

这种结构模拟了高维分类里常见的问题：很多变量彼此相关，单个变量不一定唯一重要，但一组变量共同携带相近信息。

## D2-D3. L1 与 L2 对比

我使用标准化流程，并在训练集内部划分验证集选择超参数 `C`。测试集结果如下：

{markdown_table(reg_display)}

图 `figures/regularization_performance.png` 展示性能指标：

- 横轴是指标名称，包括 accuracy、recall、ROC-AUC。
- 纵轴是指标值。
- 蓝色柱代表 L1 正则化，橙色柱代表 L2 正则化。

图 `figures/regularization_complexity.png` 展示模型复杂度：

- 左图横轴是正则化类型，纵轴是非零系数个数。
- 右图横轴是系数大小，纵轴是落在该区间内的特征数量。
- 两种颜色分别对应 L1 和 L2。

## D4. 核心问题回答

L1 和 L2 的预测表现没有出现数量级差异。它们都能在高维相关特征下得到可用的分类器；差别主要体现在复杂度和解释方式上。

更稀疏的是 L1。L1 会把一部分系数压到 0，因此更适合“给出一个更短的变量名单”。如果业务方想知道哪些变量最值得保留，L1 更直接。

如果业务方更在意模型稳定性而不是变量筛选，我会更偏向 L2。原因是 L2 不会在强相关变量之间强行只留一个，而是把权重更平滑地分配给相关变量组，预测通常更稳定。
"""

    real_metrics = real_results["metrics_05"]
    real_best = real_results["best_f1"]
    real_report = f"""# Week 15 任务E：真实数据挑战

## E1. 真实二分类数据

我使用 sklearn 内置的 Breast Cancer Wisconsin 数据集作为真实二分类任务。原始数据包含 569 个样本和 30 个数值特征。本报告把 malignant 设为正类，数据保存为 `src/week15/data/real_binary_breast_cancer.csv`。

正类比例为 `{real_results["positive_rate"]:.4f}`。这说明数据并非完全均衡，因此不能只盯着 accuracy。

## E2. 完整逻辑回归流程

流程包括：数据读取、正类定义、训练/测试划分、标准化、普通 LogisticRegression 训练、阈值扫描和图形分析。阈值 0.5 下的主要结果如下：

| 指标 | 数值 |
|:--|--:|
| accuracy | {real_metrics["accuracy"]:.4f} |
| precision | {real_metrics["precision"]:.4f} |
| recall | {real_metrics["recall"]:.4f} |
| F1 | {real_metrics["F1"]:.4f} |
| ROC-AUC | {real_metrics["ROC_AUC"]:.4f} |
| log loss | {real_metrics["log_loss"]:.4f} |

图 `figures/real_threshold_metrics.png` 展示真实数据上的阈值权衡：

- 横轴是分类阈值 threshold。
- 纵轴是指标值。
- 四条线分别是 accuracy、precision、recall、F1。

## E3. 业务解释

在这个数据里，单看 accuracy 可能误导判断。医疗筛查更怕漏诊，所以 recall 和 ROC-AUC 更值得关注。accuracy 高不代表模型没有漏掉恶性样本。

我最后更信任 recall、F1 和 ROC-AUC 的组合。ROC-AUC 反映整体排序能力，recall 反映能否抓住真正恶性病例，F1 则在 precision 和 recall 之间做折中。

如果向业务方解释模型输出，我会强调“概率”而不只是“类别”。因为概率可以配合不同阈值服务不同风险偏好：初筛阶段可以降低阈值提高召回，复核阶段可以提高阈值减少误报。本次 F1 最优阈值约为 `{real_best["threshold"]:.2f}`，对应 F1 为 `{real_best["F1"]:.4f}`。
"""

    summary = """# Week 15 总结

## 1. 为什么逻辑回归不是“线性回归后面接一个 sigmoid”这么简单？

逻辑回归表面上确实有线性部分和 sigmoid 映射，但它的统计含义不是把线性回归的预测值简单压到 0 到 1 之间。逻辑回归假设标签服从 Bernoulli 分布，模型输出的是 `P(y=1|x)`，训练目标来自 Bernoulli likelihood 的最大化。

因此，逻辑回归的核心是概率建模，而不是给线性回归补一个非线性外壳。它的系数、概率、阈值和 log loss 都服务于二分类概率解释。

## 2. sigmoid、Bernoulli likelihood 和 log loss 的关系

sigmoid 把线性预测子映射成概率。Bernoulli likelihood 用这个概率解释观测到的 0/1 标签。log loss 则是 Bernoulli likelihood 取负对数后的训练目标。

三者关系可以理解为：sigmoid 给出概率，Bernoulli likelihood 衡量这个概率对真实标签的解释能力，log loss 把最大化似然变成最小化损失。

## 3. 为什么分类模型不能只看 accuracy？

accuracy 把所有判断都合成一个比例，容易掩盖 FP 和 FN 的代价差异。在疾病初筛中，漏诊和误报的成本完全不同；在用户流失预警中，漏掉高风险用户和误打扰普通用户也不是同一回事。

所以分类模型必须结合 precision、recall、F1、ROC-AUC、log loss 以及业务阈值一起看。阈值改变后，指标会发生系统性权衡。

## 4. L1 和 L2 逻辑回归分别适合什么目标？

L1 更适合变量筛选。它能把部分系数压到 0，形成较短的变量名单，适合解释“哪些变量被保留”。

L2 更适合稳定预测。它通常不会把系数压成 0，而是平滑地惩罚大系数，在多重共线或变量组相关时更稳健。

## 5. 为什么逻辑回归仍然是强 baseline？

如果业务方需要稳定概率、可解释变量方向、可调阈值和较低实现成本，逻辑回归仍然很强。它比黑箱模型更容易解释，也比普通线性回归更符合二分类概率结构。

本周实验说明：逻辑回归不仅能给出类别，还能给出概率；不仅能配合阈值做业务权衡，还能通过 L1/L2 正则化适应高维和共线性场景。
"""

    files = {
        "synthetic_report.md": synthetic_report,
        "threshold_report.md": threshold_report,
        "regularization_report.md": regularization_report,
        "real_data_report.md": real_report,
        "summary.md": summary,
    }
    for name, content in files.items():
        (RESULTS_DIR / name).write_text(content, encoding="utf-8", newline="\n")


def main():
    synthetic_df = make_synthetic_binary()
    synthetic_df.to_csv(DATA_DIR / "synthetic_binary.csv", index=False)
    synthetic_results = run_synthetic_tasks(synthetic_df)
    reg_comparison, _ = run_regularization_task()
    real_results = run_real_data_task()
    write_reports(synthetic_df, synthetic_results, reg_comparison, real_results)

    summary = {
        "synthetic_rows": len(synthetic_df),
        "synthetic_positive_rate": rounded(float(synthetic_df["y"].mean())),
        "logistic_accuracy": rounded(synthetic_results["logistic_metrics"]["accuracy"]),
        "logistic_recall": rounded(synthetic_results["logistic_metrics"]["recall"]),
        "regularization": reg_comparison.round(4).to_dict(orient="records"),
        "real_accuracy": rounded(real_results["metrics_05"]["accuracy"]),
        "real_recall": rounded(real_results["metrics_05"]["recall"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Week 15 completed.")


if __name__ == "__main__":
    main()
