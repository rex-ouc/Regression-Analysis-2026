from pathlib import Path
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
RANDOM_SEED = 42

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for folder in [RESULTS_DIR, FIGURES_DIR]:
        for path in folder.glob("*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def md_table(df, digits=4):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(digits)
    try:
        return out.to_markdown(index=False)
    except Exception:
        return out.to_string(index=False)


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def predict_by_threshold(prob, threshold):
    return (np.asarray(prob) >= threshold).astype(int)


def logistic_pipeline(**kwargs):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**kwargs)),
        ]
    )


def generate_synthetic_binary(n=600):
    rng = np.random.default_rng(RANDOM_SEED)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    x4 = rng.normal(0, 1, n)
    eta = -0.25 + 1.8 * x1 - 1.4 * x2 + 0.7 * x3 + 0.0 * x4
    p = sigmoid(eta)
    y = rng.binomial(1, p, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "true_probability": p, "y": y})
    df.to_csv(DATA_DIR / "synthetic_binary.csv", index=False)
    return df


def run_task_a(df):
    features = ["x1", "x2", "x3", "x4"]
    X = df[features]
    y = df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    linear = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    logistic = logistic_pipeline(max_iter=3000)
    linear.fit(X_train, y_train)
    logistic.fit(X_train, y_train)

    linear_score = linear.predict(X_test)
    logistic_prob = logistic.predict_proba(X_test)[:, 1]
    linear_pred = predict_by_threshold(linear_score, 0.5)
    logistic_pred = predict_by_threshold(logistic_prob, 0.5)

    rows = []
    for name, score, pred in [
        ("LinearRegression", linear_score, linear_pred),
        ("LogisticRegression", logistic_prob, logistic_pred),
    ]:
        metrics = binary_metrics(y_test, pred)
        rows.append(
            {
                "model": name,
                **metrics,
                "ROC_AUC": roc_auc_score(y_test, score),
                "log_loss": log_loss(y_test, np.clip(score, 1e-6, 1 - 1e-6)),
            }
        )
    comparison = pd.DataFrame(rows)

    plot_linear_vs_logistic(X_train, y_train, linear, logistic)
    return {
        "features": features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "linear": linear,
        "logistic": logistic,
        "linear_score": linear_score,
        "logistic_prob": logistic_prob,
        "comparison": comparison,
    }


def plot_linear_vs_logistic(X_train, y_train, linear, logistic):
    x_grid = np.linspace(X_train["x1"].quantile(0.01), X_train["x1"].quantile(0.99), 250)
    base = X_train.mean()
    grid = pd.DataFrame([base] * len(x_grid))
    grid["x1"] = x_grid

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train["x1"], y_train, s=18, alpha=0.25, label="observed 0/1 label")
    plt.plot(x_grid, linear.predict(grid), linewidth=2, label="LinearRegression output")
    plt.plot(x_grid, logistic.predict_proba(grid)[:, 1], linewidth=2, label="LogisticRegression probability")
    plt.axhline(0, color="black", linewidth=1, alpha=0.35)
    plt.axhline(1, color="black", linewidth=1, alpha=0.35)
    plt.xlabel("x1")
    plt.ylabel("model output")
    plt.title("Linear Output vs Logistic Probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "linear_vs_logistic_output.png", dpi=180)
    plt.close()


def plot_loss_curves():
    p = np.linspace(0.001, 0.999, 500)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    axes[0].plot(p, (1 - p) ** 2, label="squared error")
    axes[0].plot(p, -np.log(p), label="log loss")
    axes[0].set_title("True label y = 1")
    axes[0].set_xlabel("predicted probability p")
    axes[0].set_ylabel("loss value")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(p, p ** 2, label="squared error")
    axes[1].plot(p, -np.log(1 - p), label="log loss")
    axes[1].set_title("True label y = 0")
    axes[1].set_xlabel("predicted probability p")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("Squared Error vs Log Loss")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "loss_curve_comparison.png", dpi=180)
    plt.close(fig)


def threshold_scan(y_true, prob):
    rows = []
    for threshold in np.arange(0.1, 1.0, 0.1):
        rows.append({"threshold": threshold, **binary_metrics(y_true, predict_by_threshold(prob, threshold))})
    threshold_df = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    for col in ["accuracy", "precision", "recall", "F1"]:
        plt.plot(threshold_df["threshold"], threshold_df[col], marker="o", label=col)
    plt.xlabel("classification threshold")
    plt.ylabel("metric value")
    plt.title("Classification Metrics Across Thresholds")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_metric_curves.png", dpi=180)
    plt.close()
    return threshold_df


def generate_high_dimensional_binary(n=700):
    rng = np.random.default_rng(RANDOM_SEED + 15)
    z = rng.normal(0, 1, n)
    data = {
        "x1": z + rng.normal(0, 0.10, n),
        "x2": 0.90 * z + rng.normal(0, 0.10, n),
        "x3": 0.85 * z + rng.normal(0, 0.12, n),
    }
    for i in range(4, 26):
        data[f"x{i}"] = rng.normal(0, 1, n)
    df = pd.DataFrame(data)
    eta = -0.1 + 1.4 * df["x1"] - 1.2 * df["x4"] + 0.9 * df["x5"] + 0.75 * df["x9"] - 0.65 * df["x12"]
    df["y"] = rng.binomial(1, sigmoid(eta), n)
    return df


def run_regularization_task():
    df = generate_high_dimensional_binary()
    features = [c for c in df.columns if c != "y"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["y"], test_size=0.3, random_state=RANDOM_SEED, stratify=df["y"]
    )
    param_grid = {"model__C": np.logspace(-2, 2, 12)}
    configs = {
        "L1": logistic_pipeline(penalty="l1", solver="liblinear", max_iter=5000),
        "L2": logistic_pipeline(penalty="l2", solver="liblinear", max_iter=5000),
    }
    rows = []
    coef_rows = []
    for name, estimator in configs.items():
        grid = GridSearchCV(estimator, param_grid, cv=5, scoring="neg_log_loss")
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        prob = model.predict_proba(X_test)[:, 1]
        pred = predict_by_threshold(prob, 0.5)
        metrics = binary_metrics(y_test, pred)
        coef = model.named_steps["model"].coef_[0]
        rows.append(
            {
                "model": name,
                "best_C": grid.best_params_["model__C"],
                "accuracy": metrics["accuracy"],
                "recall": metrics["recall"],
                "ROC_AUC": roc_auc_score(y_test, prob),
                "log_loss": log_loss(y_test, prob),
                "nonzero_coefficients": int(np.sum(np.abs(coef) > 1e-6)),
            }
        )
        for feature, value in zip(features, coef):
            coef_rows.append({"model": name, "feature": feature, "coefficient": value, "abs_coefficient": abs(value)})

    perf = pd.DataFrame(rows)
    coefs = pd.DataFrame(coef_rows)
    plot_regularization_figures(perf)
    return {"performance": perf, "coefs": coefs}


def plot_regularization_figures(perf):
    metrics = ["accuracy", "recall", "ROC_AUC"]
    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(8, 5))
    for i, model in enumerate(perf["model"]):
        vals = perf.loc[perf["model"] == model, metrics].iloc[0]
        plt.bar(x + (i - 0.5) * width, vals, width, label=model)
    plt.xticks(x, metrics)
    plt.ylabel("metric value")
    plt.title("L1 vs L2 Logistic Regression Performance")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "regularization_performance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4.5))
    plt.bar(perf["model"], perf["nonzero_coefficients"])
    plt.xlabel("regularization type")
    plt.ylabel("number of nonzero coefficients")
    plt.title("Model Complexity: Nonzero Coefficients")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "regularization_complexity.png", dpi=180)
    plt.close()


def write_synthetic_report(df, task):
    out_of_range = int(np.sum((task["linear_score"] < 0) | (task["linear_score"] > 1)))
    content = f"""# 第 15 周：模拟二分类数据报告

## 1. 数据生成机制

本次模拟数据一共有 `{len(df)}` 个样本，建模时使用 `4` 个特征。

目标变量不是直接用硬阈值手工生成的，而是先生成每个样本属于正类的概率，再从 Bernoulli 分布中抽样得到 `0/1` 标签：

```text
eta = -0.25 + 1.8*x1 - 1.4*x2 + 0.7*x3 + 0.0*x4
p = 1 / (1 + exp(-eta))
y ~ Bernoulli(p)
```

其中，`x1` 和 `x3` 的系数为正，会提高样本属于正类的概率；`x2` 的系数为负，会降低样本属于正类的概率；`x4` 的真实系数为 0，是一个噪声变量。

## 2. LinearRegression 与 LogisticRegression 对比

{md_table(task['comparison'])}

在测试集中，LinearRegression 有 `{out_of_range}` 个输出落在 `[0, 1]` 之外。虽然它也可以通过 0.5 阈值被强行用于分类，但它的输出本身不是自然的概率。

LogisticRegression 会把线性得分经过 sigmoid 函数转换，因此输出一定在 0 到 1 之间，可以解释为样本属于正类的估计概率。

## 3. 图片解释：`linear_vs_logistic_output.png`

这张图的横轴是特征 `x1`，纵轴是模型输出。散点表示训练集中真实观测到的 `0/1` 标签；LinearRegression 的线表示线性回归的原始输出；LogisticRegression 的曲线表示预测为正类的概率。

这张图想说明的核心不是“能不能分类”，因为两个模型都可以设阈值分类。真正重要的是：LogisticRegression 的输出具有概率含义，而 LinearRegression 的输出不一定能被解释为概率。
"""
    (RESULTS_DIR / "synthetic_report.md").write_text(content, encoding="utf-8")


def write_threshold_report(task, threshold_df):
    base = pd.DataFrame([{"threshold": 0.5, **binary_metrics(task["y_test"], predict_by_threshold(task["logistic_prob"], 0.5))}])
    content = f"""# 第 15 周：阈值分析与 Log Loss 报告

## 1. Bernoulli 分布

$$Y \\sim Bernoulli(p)$$

在二分类问题中，目标变量只有两个可能取值：0 或 1。参数 `p` 表示样本属于正类，也就是 `y = 1` 的概率。这个设定和逻辑回归非常匹配，因为逻辑回归输出的也是正类概率。

## 2. 单个样本的似然函数

$$L(p;y)=p^y(1-p)^{{1-y}}$$

这个公式把 `y = 1` 和 `y = 0` 两种情况写在了一起。如果真实标签 `y = 1`，似然就是 `p`；如果真实标签 `y = 0`，似然就是 `1-p`。一个好的概率模型，应该给真实发生的类别更高概率。

## 3. 负对数似然与 Log Loss

$$-\\log L(p;y)=-[y\\log(p)+(1-y)\\log(1-p)]$$

这就是逻辑回归常用的 log loss。模型预测正确而且很自信时，损失会很小；模型预测错误但非常自信时，损失会迅速变大。这符合二分类概率建模的直觉。

## 4. 图片解释：`loss_curve_comparison.png`

这张图的横轴是模型预测为正类的概率 `p`，纵轴是损失值。左图固定真实标签为 `y = 1`，右图固定真实标签为 `y = 0`。每个子图都比较了 squared error 和 log loss 两种损失。

图中可以看到，当模型“错得很自信”时，log loss 的惩罚比平方误差更重。因此，一旦我们把模型输出解释成概率，log loss 就比 MSE 更自然。

## 5. 阈值为 0.5 时的混淆矩阵与基础指标

{md_table(base)}

## 6. 阈值扫描结果

{md_table(threshold_df)}

## 7. 图片解释：`threshold_metric_curves.png`

这张图的横轴是分类阈值，纵轴是指标数值。图中的四条曲线分别代表 accuracy、precision、recall 和 F1。

当阈值升高时，模型会更少预测正类。通常 precision 可能上升，因为模型变得更保守；recall 往往下降，因为更多真实正类会被漏掉。

如果业务场景是疾病初筛，我最关心 recall。因为漏掉一个真正患病的人，代价通常高于让健康人多做一次复查。所以如果让我推荐阈值，我会选择相对较低的阈值，以减少假阴性。
"""
    (RESULTS_DIR / "threshold_report.md").write_text(content, encoding="utf-8")


def write_regularization_report(reg):
    perf = reg["performance"]
    coefs = reg["coefs"]
    l1_nonzero = int(perf.loc[perf["model"] == "L1", "nonzero_coefficients"].iloc[0])
    l2_nonzero = int(perf.loc[perf["model"] == "L2", "nonzero_coefficients"].iloc[0])
    top_l1 = coefs[coefs["model"] == "L1"].sort_values("abs_coefficient", ascending=False).head(10)
    content = f"""# 第 15 周：L1 与 L2 正则化逻辑回归报告

## 1. 高维二分类数据

本任务中，我生成了一份包含 25 个特征的二分类数据。`x1`、`x2` 和 `x3` 来自同一个潜在变量，因此它们之间存在明显相关性。部分变量真实影响类别概率，其余变量主要是噪声变量。

L1 和 L2 模型都放在 Pipeline 中，先进行 StandardScaler 标准化，再训练 LogisticRegression。超参数 `C` 通过 5 折 GridSearchCV 选择，评价标准是 negative log loss。`C` 越小，正则化越强。

## 2. 性能与模型复杂度

{md_table(perf)}

## 3. 图片解释：`regularization_performance.png`

这张图的横轴是评价指标，包括 accuracy、recall 和 ROC-AUC；纵轴是指标数值。不同颜色的柱子分别代表 L1 和 L2 正则化逻辑回归。它主要用于比较两种正则化方式的预测表现。

## 4. 图片解释：`regularization_complexity.png`

这张图的横轴是正则化类型，纵轴是非零系数的数量。它主要用于比较模型复杂度。

L1 正则化通常会得到更稀疏的模型，因为它可以把一部分系数直接压缩为 0。L2 正则化通常会保留更多变量，但会平滑地缩小系数大小。

## 5. L1 模型中系数绝对值最大的变量

{md_table(top_l1[['feature', 'coefficient', 'abs_coefficient']])}

## 6. 核心比较

在本次实验中，L1 和 L2 的预测表现差距并不是特别大，更明显的差异体现在模型稀疏性上。

L1 保留了 `{l1_nonzero}` 个非零系数，而 L2 保留了 `{l2_nonzero}` 个非零系数。如果业务目标是给出更短的变量名单，我会更倾向于 L1；如果业务方更在意模型稳定性，尤其是在变量相关性较强的情况下，我会更倾向于 L2。
"""
    (RESULTS_DIR / "regularization_report.md").write_text(content, encoding="utf-8")


def write_summary(task, reg):
    best = task["comparison"].loc[task["comparison"]["ROC_AUC"].idxmax(), "model"]
    perf = reg["performance"]
    l1_nonzero = int(perf.loc[perf["model"] == "L1", "nonzero_coefficients"].iloc[0])
    l2_nonzero = int(perf.loc[perf["model"] == "L2", "nonzero_coefficients"].iloc[0])
    content = f"""# 第 15 周总结

## 1. 为什么逻辑回归不只是“线性回归加 sigmoid”

逻辑回归和线性回归的统计假设不同。逻辑回归假设目标变量服从 Bernoulli 分布，用 sigmoid 把线性得分映射为概率，并通过最大化 Bernoulli likelihood 来估计参数。这个过程自然会导出 log loss。

在模拟数据实验中，ROC-AUC 更高的模型是 `{best}`。

## 2. Sigmoid、Bernoulli Likelihood 与 Log Loss 的关系

sigmoid 函数把任意实数线性得分转换成 0 到 1 之间的概率。Bernoulli likelihood 衡量在这个概率下，真实观测到的 0/1 标签有多可能发生。对 likelihood 取负对数，就得到 log loss。

## 3. 为什么分类模型不能只看 Accuracy

accuracy 会掩盖 false positive 和 false negative 的差异。阈值分析说明，同一个概率模型在不同阈值下会得到不同的 precision 和 recall。因此，分类任务不能只看 accuracy，指标选择必须结合业务场景。

## 4. L1 与 L2 逻辑回归分别适合什么目标

L1 更适合变量筛选和给出较短的特征名单。L2 更适合在变量相关性较强时追求稳定预测，因为它倾向于平滑收缩系数，而不是直接删除大量变量。

在本次实验中，L1 保留了 `{l1_nonzero}` 个非零系数，L2 保留了 `{l2_nonzero}` 个非零系数。

## 5. 为什么逻辑回归仍然是很强的 baseline

逻辑回归可以输出概率，支持根据业务目标调整阈值，系数方向也比较容易解释。同时，它还能通过 L1 或 L2 正则化处理高维数据。因此在很多二分类任务中，逻辑回归仍然是一个透明、稳定、很有价值的 baseline 模型。
"""
    (RESULTS_DIR / "summary.md").write_text(content, encoding="utf-8")


def write_real_data_report():
    content = """# 第 15 周：真实数据报告

真实数据任务是选做题。本次提交主要完成必做部分：模拟二分类数据实验、log loss 理论解释、阈值扫描分析，以及 L1/L2 正则化逻辑回归对比。
"""
    (RESULTS_DIR / "real_data_report.md").write_text(content, encoding="utf-8")


def main():
    ensure_dirs()
    print("===== Week 15 Logistic Regression Started =====")

    print("[1] Generating synthetic binary data")
    df = generate_synthetic_binary()

    print("[2] Comparing LinearRegression and LogisticRegression")
    task = run_task_a(df)

    print("[3] Plotting log loss curves and scanning thresholds")
    plot_loss_curves()
    threshold_df = threshold_scan(task["y_test"], task["logistic_prob"])

    print("[4] Comparing L1 and L2 regularized logistic regression")
    reg = run_regularization_task()

    print("[5] Writing reports")
    write_synthetic_report(df, task)
    write_threshold_report(task, threshold_df)
    write_regularization_report(reg)
    write_summary(task, reg)
    write_real_data_report()

    print("===== Week 15 Finished =====")
    print("Data:", DATA_DIR / "synthetic_binary.csv")
    print("Figures:", FIGURES_DIR)
    print("Reports:", RESULTS_DIR)


if __name__ == "__main__":
    main()
