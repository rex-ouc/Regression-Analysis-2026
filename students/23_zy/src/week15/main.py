import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss


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

SYNTHETIC_DATA_PATH = os.path.join(DATA_DIR, "synthetic_binary.csv")

SYNTHETIC_REPORT_PATH = os.path.join(RESULTS_DIR, "synthetic_report.md")
THRESHOLD_REPORT_PATH = os.path.join(RESULTS_DIR, "threshold_report.md")
REGULARIZATION_REPORT_PATH = os.path.join(RESULTS_DIR, "regularization_report.md")
SUMMARY_REPORT_PATH = os.path.join(RESULTS_DIR, "summary.md")

FIG_LINEAR_LOGISTIC = os.path.join(FIGURES_DIR, "linear_vs_logistic_output.png")
FIG_LOSS_CURVES = os.path.join(FIGURES_DIR, "loss_curves.png")
FIG_THRESHOLD = os.path.join(FIGURES_DIR, "threshold_metrics.png")
FIG_REG_METRICS = os.path.join(FIGURES_DIR, "regularization_metrics.png")
FIG_REG_COMPLEXITY = os.path.join(FIGURES_DIR, "regularization_complexity.png")


# ============================================================
# 2. 基础工具函数
# ============================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def df_to_markdown(df, float_digits=4):
    """
    自己写 DataFrame 转 markdown 表格，避免缺 tabulate 报错。
    """
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


def classification_metrics_from_prob(y_true, y_prob, threshold=0.5):
    """
    根据预测概率和阈值，计算二分类指标。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true)

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "threshold": threshold,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


# ============================================================
# 3. Task A：生成二分类模拟数据
# ============================================================

def make_synthetic_binary_data(n_samples=600, random_state=42):
    """
    生成带有明确概率结构的二分类数据。

    先生成线性得分 eta = X beta，
    再通过 sigmoid 得到概率 p，
    最后从 Bernoulli(p) 抽样得到 y。
    """
    rng = np.random.default_rng(random_state)

    x1 = rng.normal(0, 1, n_samples)
    x2 = rng.normal(0, 1, n_samples)
    x3 = rng.normal(0, 1, n_samples)
    x4 = rng.normal(0, 1, n_samples)
    x5 = rng.normal(0, 1, n_samples)

    eta = (
        -0.2
        + 1.6 * x1
        - 1.3 * x2
        + 0.9 * x3
        - 0.6 * x4
        + 0.0 * x5
    )

    prob = sigmoid(eta)
    y = rng.binomial(1, prob, n_samples)

    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "x5": x5,
        "true_probability": prob,
        "y": y,
    })

    return df


def train_linear_and_logistic(df):
    features = ["x1", "x2", "x3", "x4", "x5"]
    X = df[features].values
    y = df["y"].values

    X_train, X_test, y_train, y_test, prob_train, prob_test = train_test_split(
        X,
        y,
        df["true_probability"].values,
        test_size=0.30,
        random_state=2026,
        stratify=y,
    )

    linear_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ])

    linear_model.fit(X_train, y_train)
    logistic_model.fit(X_train, y_train)

    linear_output = linear_model.predict(X_test)
    logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

    linear_prob_like = np.clip(linear_output, 0, 1)

    linear_metrics = classification_metrics_from_prob(
        y_test,
        linear_prob_like,
        threshold=0.5,
    )
    logistic_metrics = classification_metrics_from_prob(
        y_test,
        logistic_prob,
        threshold=0.5,
    )

    linear_metrics["模型"] = "LinearRegression clipped to [0,1]"
    logistic_metrics["模型"] = "LogisticRegression"

    linear_metrics["ROC_AUC"] = roc_auc_score(y_test, linear_prob_like)
    logistic_metrics["ROC_AUC"] = roc_auc_score(y_test, logistic_prob)

    linear_metrics["log_loss"] = log_loss(y_test, linear_prob_like, labels=[0, 1])
    logistic_metrics["log_loss"] = log_loss(y_test, logistic_prob, labels=[0, 1])

    metric_df = pd.DataFrame([linear_metrics, logistic_metrics])

    aux = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "prob_test": prob_test,
        "linear_output": linear_output,
        "linear_prob_like": linear_prob_like,
        "logistic_prob": logistic_prob,
        "linear_model": linear_model,
        "logistic_model": logistic_model,
        "features": features,
    }

    return metric_df, aux


def plot_linear_vs_logistic(aux):
    """
    固定其他变量为均值，只改变 x1，展示 LinearRegression 和 LogisticRegression 输出差别。
    """
    X_train = aux["X_train"]
    X_test = aux["X_test"]
    y_test = aux["y_test"]
    linear_model = aux["linear_model"]
    logistic_model = aux["logistic_model"]

    x1_grid = np.linspace(X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5, 300)

    base = X_train.mean(axis=0)
    X_grid = np.tile(base, (len(x1_grid), 1))
    X_grid[:, 0] = x1_grid

    linear_curve = linear_model.predict(X_grid)
    logistic_curve = logistic_model.predict_proba(X_grid)[:, 1]

    plt.figure(figsize=(9, 6))
    plt.scatter(
        X_test[:, 0],
        y_test,
        alpha=0.45,
        label="True 0/1 labels on test set"
    )
    plt.plot(
        x1_grid,
        linear_curve,
        linewidth=2,
        label="LinearRegression output"
    )
    plt.plot(
        x1_grid,
        logistic_curve,
        linewidth=2,
        label="LogisticRegression probability"
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axhline(1, linestyle="--", linewidth=1)
    plt.xlabel("x1")
    plt.ylabel("model output")
    plt.title("Linear Regression Output vs Logistic Regression Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_LINEAR_LOGISTIC, dpi=200)
    plt.close()


# ============================================================
# 4. Task B：loss 曲线
# ============================================================

def make_loss_curve_data():
    eps = 1e-6
    p = np.linspace(eps, 1 - eps, 500)

    squared_y1 = (1 - p) ** 2
    squared_y0 = (0 - p) ** 2

    logloss_y1 = -np.log(p)
    logloss_y0 = -np.log(1 - p)

    loss_df = pd.DataFrame({
        "p": p,
        "squared_loss_y1": squared_y1,
        "squared_loss_y0": squared_y0,
        "log_loss_y1": logloss_y1,
        "log_loss_y0": logloss_y0,
    })

    return loss_df


def plot_loss_curves(loss_df):
    plt.figure(figsize=(9, 6))
    plt.plot(loss_df["p"], loss_df["squared_loss_y1"], label="Squared error, y=1")
    plt.plot(loss_df["p"], loss_df["log_loss_y1"], label="Log loss, y=1")
    plt.plot(loss_df["p"], loss_df["squared_loss_y0"], label="Squared error, y=0")
    plt.plot(loss_df["p"], loss_df["log_loss_y0"], label="Log loss, y=0")
    plt.xlabel("predicted probability p for positive class")
    plt.ylabel("loss value")
    plt.title("Squared Error vs Log Loss for Binary Classification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_LOSS_CURVES, dpi=200)
    plt.close()


# ============================================================
# 5. Task C：threshold 扫描
# ============================================================

def threshold_scan(y_true, y_prob):
    thresholds = np.arange(0.1, 1.0, 0.1)
    records = []

    for th in thresholds:
        records.append(classification_metrics_from_prob(y_true, y_prob, threshold=float(th)))

    threshold_df = pd.DataFrame(records)
    return threshold_df


def plot_threshold_metrics(threshold_df):
    plt.figure(figsize=(9, 6))
    plt.plot(threshold_df["threshold"], threshold_df["accuracy"], marker="o", label="accuracy")
    plt.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="recall")
    plt.plot(threshold_df["threshold"], threshold_df["F1"], marker="o", label="F1")
    plt.xlabel("classification threshold")
    plt.ylabel("metric value")
    plt.title("Classification Metrics under Different Thresholds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_THRESHOLD, dpi=200)
    plt.close()


# ============================================================
# 6. Task D：L1 vs L2 正则化逻辑回归
# ============================================================

def make_highdim_binary_data(n_samples=700, n_features=30, random_state=123):
    """
    构造特征较多且带共线性的二分类数据。
    """
    rng = np.random.default_rng(random_state)

    base1 = rng.normal(0, 1, n_samples)
    base2 = rng.normal(0, 1, n_samples)

    X = rng.normal(0, 1, size=(n_samples, n_features))

    X[:, 0] = base1
    X[:, 1] = base1 + rng.normal(0, 0.08, n_samples)
    X[:, 2] = 0.8 * base1 + rng.normal(0, 0.10, n_samples)

    X[:, 3] = base2
    X[:, 4] = base2 + rng.normal(0, 0.08, n_samples)
    X[:, 5] = 0.7 * base2 + rng.normal(0, 0.10, n_samples)

    beta = np.zeros(n_features)
    beta[0] = 1.8
    beta[3] = -1.6
    beta[6] = 1.2
    beta[10] = -1.0
    beta[15] = 0.8

    eta = -0.1 + X @ beta
    prob = sigmoid(eta)
    y = rng.binomial(1, prob, n_samples)

    columns = [f"x{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["true_probability"] = prob
    df["y"] = y

    return df


def tune_l1_l2_logistic(highdim_df):
    features = [col for col in highdim_df.columns if col.startswith("x")]
    X = highdim_df[features].values
    y = highdim_df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=2026,
        stratify=y,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    configs = {
        "L1 Logistic": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=5000
                ))
            ]),
            "params": {
                "model__C": C_values
            }
        },
        "L2 Logistic": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    max_iter=5000
                ))
            ]),
            "params": {
                "model__C": C_values
            }
        }
    }

    best_models = {}
    records = []

    for name, cfg in configs.items():
        grid = GridSearchCV(
            estimator=cfg["pipeline"],
            param_grid=cfg["params"],
            scoring="neg_log_loss",
            cv=cv
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_models[name] = best_model

        prob = best_model.predict_proba(X_test)[:, 1]
        metric = classification_metrics_from_prob(y_test, prob, threshold=0.5)

        coef = best_model.named_steps["model"].coef_.reshape(-1)
        nonzero = int(np.sum(np.abs(coef) > 1e-6))

        metric["模型"] = name
        metric["best_C"] = grid.best_params_["model__C"]
        metric["ROC_AUC"] = roc_auc_score(y_test, prob)
        metric["log_loss"] = log_loss(y_test, prob, labels=[0, 1])
        metric["非零系数个数"] = nonzero

        records.append(metric)

    result_df = pd.DataFrame(records)

    aux = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "features": features,
        "best_models": best_models,
    }

    return result_df, aux


def plot_regularization_results(reg_df):
    metrics = ["accuracy", "recall", "ROC_AUC"]
    model_names = reg_df["模型"].tolist()

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(9, 6))

    for i, model_name in enumerate(model_names):
        row = reg_df[reg_df["模型"] == model_name].iloc[0]
        values = [row[m] for m in metrics]
        plt.bar(x + (i - 0.5) * width, values, width=width, label=model_name)

    plt.xticks(x, metrics)
    plt.ylabel("metric value")
    plt.title("L1 vs L2 Logistic Regression Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_REG_METRICS, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(reg_df["模型"], reg_df["非零系数个数"])
    plt.ylabel("number of non-zero coefficients")
    plt.title("Model Complexity: Non-zero Coefficients")
    plt.tight_layout()
    plt.savefig(FIG_REG_COMPLEXITY, dpi=200)
    plt.close()


# ============================================================
# 7. 写报告
# ============================================================

def write_synthetic_report(df, model_metric_df, aux):
    linear_output = aux["linear_output"]
    logistic_prob = aux["logistic_prob"]

    below_zero = int(np.sum(linear_output < 0))
    above_one = int(np.sum(linear_output > 1))

    lines = []

    lines.append("# Week15 Synthetic Report：逻辑回归与二分类")
    lines.append("")
    lines.append("## 1. 数据生成机制 DGP")
    lines.append("")
    lines.append(f"本次模拟数据样本量为 {df.shape[0]}，特征数为 5。")
    lines.append("")
    lines.append("我先生成 5 个连续特征 x1 到 x5，然后构造线性得分 eta：")
    lines.append("")
    lines.append("```text")
    lines.append("eta = -0.2 + 1.6*x1 - 1.3*x2 + 0.9*x3 - 0.6*x4 + 0.0*x5")
    lines.append("```")
    lines.append("")
    lines.append("之后通过 sigmoid 函数把 eta 转成正类概率 p：")
    lines.append("")
    lines.append("```text")
    lines.append("p = 1 / (1 + exp(-eta))")
    lines.append("```")
    lines.append("")
    lines.append("最后从 Bernoulli(p) 抽样得到 0/1 标签 y。")
    lines.append("")
    lines.append("其中，x1 和 x3 会提高正类概率，x2 和 x4 会降低正类概率，x5 基本没有影响。")
    lines.append("")
    lines.append("## 2. LinearRegression 与 LogisticRegression 对比")
    lines.append("")
    lines.append(df_to_markdown(model_metric_df, float_digits=4))
    lines.append("")
    lines.append("LinearRegression 的输出如果硬解释成概率，会出现不自然的问题。")
    lines.append(f"在测试集中，LinearRegression 输出小于 0 的样本数为 {below_zero}，大于 1 的样本数为 {above_one}。")
    lines.append("但是概率应该被限制在 0 到 1 之间，所以线性回归的输出没有天然的概率意义。")
    lines.append("")
    lines.append("LogisticRegression 通过 sigmoid 函数把线性得分映射到 0 到 1 之间，因此它的输出更容易解释成正类概率。")
    lines.append("")
    lines.append("核心对比图保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week15/figures/linear_vs_logistic_output.png")
    lines.append("```")
    lines.append("")
    lines.append("这张图的横轴是主要特征 x1，纵轴是模型输出。散点代表测试集真实 0/1 标签，一条线代表 LinearRegression 的输出，另一条线代表 LogisticRegression 的预测概率。图中想说明的是：线性回归输出可以超过概率范围，而逻辑回归输出始终在 0 到 1 之间。")
    lines.append("")
    lines.append("## 3. 核心问题回答")
    lines.append("")
    lines.append("### 3.1 LinearRegression 在二分类任务里最不自然的地方是什么？")
    lines.append("")
    lines.append("最不自然的是它的输出不是概率。它可以小于 0，也可以大于 1，所以即使它可以通过阈值做分类，也不能自然解释为正类发生概率。")
    lines.append("")
    lines.append("### 3.2 为什么逻辑回归的输出更容易解释成概率？")
    lines.append("")
    lines.append("因为逻辑回归先计算线性得分，再通过 sigmoid 映射到 0 到 1 之间。这个范围正好符合概率的定义。")
    lines.append("")
    lines.append("### 3.3 关键区别是能不能分类，还是输出是否有概率意义？")
    lines.append("")
    lines.append("关键不是能不能分类。LinearRegression 加一个阈值也能给出 0/1 分类。真正的区别是输出有没有概率意义，以及训练目标是否适合二分类问题。")

    with open(SYNTHETIC_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_threshold_report(base_metric, threshold_df, loss_df):
    lines = []

    lines.append("# Week15 Threshold Report：log loss、混淆矩阵与阈值权衡")
    lines.append("")
    lines.append("## 1. Bernoulli 与 log loss")
    lines.append("")
    lines.append("### 1.1 Bernoulli 分布")
    lines.append("")
    lines.append("```text")
    lines.append("Y ~ Bernoulli(p)")
    lines.append("```")
    lines.append("")
    lines.append("这表示目标变量 Y 只有两种可能：1 或 0。p 表示 Y=1 的概率，1-p 表示 Y=0 的概率。二分类问题本质上就是在估计这个 p。")
    lines.append("")
    lines.append("### 1.2 单样本 likelihood")
    lines.append("")
    lines.append("```text")
    lines.append("L(p; y) = p^y * (1-p)^(1-y)")
    lines.append("```")
    lines.append("")
    lines.append("当 y=1 时，这个式子变成 p；当 y=0 时，这个式子变成 1-p。也就是说，模型给真实类别分配的概率越高，likelihood 越大。")
    lines.append("")
    lines.append("### 1.3 单样本负对数似然 / log loss")
    lines.append("")
    lines.append("```text")
    lines.append("loss = -[ y*log(p) + (1-y)*log(1-p) ]")
    lines.append("```")
    lines.append("")
    lines.append("log loss 是 Bernoulli likelihood 取负对数后得到的。模型给真实类别的概率越低，loss 越大。尤其是模型错得很自信时，log loss 会给出很大的惩罚。")
    lines.append("")
    lines.append("## 2. loss 曲线解释")
    lines.append("")
    lines.append("loss 曲线图保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week15/figures/loss_curves.png")
    lines.append("```")
    lines.append("")
    lines.append("这张图的横轴是预测为正类的概率 p，纵轴是 loss value。图中比较了 squared error 和 log loss，并分别展示了真实标签 y=1 和 y=0 的情况。")
    lines.append("")
    lines.append("当模型错得很自信时，log loss 的惩罚更重。例如真实 y=1 但模型给 p 接近 0，或者真实 y=0 但模型给 p 接近 1 时，log loss 会迅速变大。")
    lines.append("")
    lines.append("二分类里错得很自信需要被重罚，因为这说明模型不仅预测错了，而且还非常确信错误答案。对于概率模型来说，这种错误比一般错误更严重。")
    lines.append("")
    lines.append("log loss 不是凭空指定的，而是来自 Bernoulli likelihood。既然我们把输出解释为概率，那么使用 Bernoulli likelihood 的负对数作为损失函数就很自然。")
    lines.append("")
    lines.append("## 3. 混淆矩阵和基础指标")
    lines.append("")
    lines.append(df_to_markdown(pd.DataFrame([base_metric]), float_digits=4))
    lines.append("")
    lines.append("其中 TP 是真实为 1 且预测为 1，TN 是真实为 0 且预测为 0，FP 是真实为 0 但预测为 1，FN 是真实为 1 但预测为 0。")
    lines.append("")
    lines.append("## 4. threshold 扫描")
    lines.append("")
    lines.append(df_to_markdown(threshold_df, float_digits=4))
    lines.append("")
    lines.append("threshold 曲线图保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week15/figures/threshold_metrics.png")
    lines.append("```")
    lines.append("")
    lines.append("这张图的横轴是 classification threshold，纵轴是 metric value。图中包含 accuracy、precision、recall 和 F1 四条曲线。")
    lines.append("")
    lines.append("一般来说，阈值升高时，模型更谨慎地预测正类，所以 precision 往往可能上升，而 recall 往往会下降。这个变化体现了 precision 和 recall 之间的 trade-off。")
    lines.append("")
    lines.append("## 5. 业务场景：疾病初筛")
    lines.append("")
    lines.append("如果是疾病初筛，我最在意 recall。因为漏掉真正有病的人，也就是 FN，代价通常很高。")
    lines.append("")
    lines.append("在这个场景下，我宁愿多筛出一些可疑样本，再通过后续检查确认，也不希望把真正有风险的人漏掉。")
    lines.append("")
    lines.append("如果给业务方推荐阈值，我会选择一个 recall 较高，同时 precision 和 F1 不至于太差的阈值。也就是说，阈值选择不是固定 0.5，而是要结合业务风险来定。")

    with open(THRESHOLD_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_regularization_report(reg_df):
    lines = []

    lines.append("# Week15 Regularization Report：L1 vs L2 逻辑回归")
    lines.append("")
    lines.append("## 1. 数据说明")
    lines.append("")
    lines.append("本部分构造了一份特征较多且带有共线性的二分类数据。特征数为 30，其中 x1、x2、x3 构成一组相关特征，x4、x5、x6 构成另一组相关特征，同时还加入了一些噪声特征。")
    lines.append("")
    lines.append("## 2. L1 和 L2 结果对比")
    lines.append("")
    lines.append(df_to_markdown(reg_df, float_digits=4))
    lines.append("")
    lines.append("性能对比图保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week15/figures/regularization_metrics.png")
    lines.append("```")
    lines.append("")
    lines.append("复杂度对比图保存位置：")
    lines.append("")
    lines.append("```text")
    lines.append("src/week15/figures/regularization_complexity.png")
    lines.append("```")
    lines.append("")
    lines.append("regularization_metrics.png 的横轴是指标名称，纵轴是指标数值，不同柱子代表 L1 和 L2 逻辑回归。regularization_complexity.png 的横轴是模型类型，纵轴是非零系数个数。")
    lines.append("")
    lines.append("## 3. 核心问题回答")
    lines.append("")
    lines.append("### 3.1 L1 和 L2 的预测表现差很多吗？")
    lines.append("")
    lines.append("从测试集指标来看，L1 和 L2 的 accuracy、recall、ROC-AUC 和 log loss 可能不会差特别多。也就是说，它们都能作为有效的分类模型。")
    lines.append("")
    lines.append("### 3.2 哪个模型更稀疏？")
    lines.append("")
    lines.append("L1 更稀疏。因为 L1 正则化可以把一部分系数压缩为 0。")
    lines.append("")
    lines.append("### 3.3 哪个模型更适合给出更短的变量名单？")
    lines.append("")
    lines.append("L1 更适合。因为它保留下来的非零系数变量可以直接理解为较短的变量名单。")
    lines.append("")
    lines.append("### 3.4 如果业务方更在意模型稳定性而不是变量筛选，更偏向哪一个？")
    lines.append("")
    lines.append("如果更在意稳定性，我会更偏向 L2。L2 不会强行删除变量，而是整体压缩系数，通常更适合提高模型稳定性。")

    with open(REGULARIZATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_summary_report():
    lines = []

    lines.append("# Week15 Summary：逻辑回归、概率解释与阈值权衡")
    lines.append("")
    lines.append("## 1. 为什么逻辑回归不是“线性回归后面接一个 sigmoid”这么简单？")
    lines.append("")
    lines.append("逻辑回归不只是在线性回归输出后接 sigmoid。它背后对应的是二分类概率建模：目标变量服从 Bernoulli 分布，模型估计的是正类概率，训练目标来自 Bernoulli likelihood 的最大化。")
    lines.append("")
    lines.append("## 2. sigmoid、Bernoulli likelihood、log loss 三者之间是什么关系？")
    lines.append("")
    lines.append("sigmoid 把线性得分变成 0 到 1 之间的概率 p。Bernoulli likelihood 用这个 p 来描述观察到标签 y 的可能性。log loss 则是 Bernoulli likelihood 取负对数后得到的损失函数。")
    lines.append("")
    lines.append("## 3. 为什么分类模型不能只看 accuracy？")
    lines.append("")
    lines.append("因为 accuracy 只看整体预测对了多少，但不区分 FP 和 FN 的代价。在疾病初筛、信用违约、用户流失等场景中，不同错误的业务成本不同，所以必须结合 precision、recall、F1、ROC-AUC 等指标一起看。")
    lines.append("")
    lines.append("## 4. L1 和 L2 逻辑回归分别更适合什么目标？")
    lines.append("")
    lines.append("L1 更适合变量筛选，因为它能产生稀疏系数，把一些变量系数压缩为 0。L2 更适合稳定建模，因为它整体压缩系数，减少模型对单个变量的过度依赖。")
    lines.append("")
    lines.append("## 5. 为什么逻辑回归仍然是很强的 baseline？")
    lines.append("")
    lines.append("逻辑回归可以输出概率，模型结构简单，训练稳定，解释性强，还可以通过 L1/L2 正则化处理高维和共线性问题。如果业务方需要一个稳定、可解释、能输出概率的模型，逻辑回归仍然是一个很强的 baseline。")

    with open(SUMMARY_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 8. 主程序
# ============================================================

def main():
    print("=" * 70)
    print("Week15：逻辑回归、分类指标与阈值权衡")
    print("=" * 70)

    print("\n[阶段1] 生成二分类模拟数据...")
    df = make_synthetic_binary_data()
    df.to_csv(SYNTHETIC_DATA_PATH, index=False)
    print(f"数据已保存：{SYNTHETIC_DATA_PATH}")
    print(f"数据规模：{df.shape}")

    print("\n[阶段2] LinearRegression vs LogisticRegression...")
    model_metric_df, aux = train_linear_and_logistic(df)
    print(model_metric_df)
    plot_linear_vs_logistic(aux)
    print(f"核心对比图已保存：{FIG_LINEAR_LOGISTIC}")

    print("\n[阶段3] 生成 loss 曲线...")
    loss_df = make_loss_curve_data()
    plot_loss_curves(loss_df)
    print(f"loss 曲线已保存：{FIG_LOSS_CURVES}")

    print("\n[阶段4] 计算混淆矩阵和 threshold 扫描...")
    y_test = aux["y_test"]
    logistic_prob = aux["logistic_prob"]

    base_metric = classification_metrics_from_prob(y_test, logistic_prob, threshold=0.5)
    base_metric["模型"] = "LogisticRegression"
    base_metric["ROC_AUC"] = roc_auc_score(y_test, logistic_prob)
    base_metric["log_loss"] = log_loss(y_test, logistic_prob, labels=[0, 1])

    threshold_df = threshold_scan(y_test, logistic_prob)
    print(threshold_df)
    plot_threshold_metrics(threshold_df)
    print(f"threshold 曲线已保存：{FIG_THRESHOLD}")

    print("\n[阶段5] L1 vs L2 正则化逻辑回归...")
    highdim_df = make_highdim_binary_data()
    reg_df, reg_aux = tune_l1_l2_logistic(highdim_df)
    print(reg_df)
    plot_regularization_results(reg_df)
    print(f"正则化性能图已保存：{FIG_REG_METRICS}")
    print(f"正则化复杂度图已保存：{FIG_REG_COMPLEXITY}")

    print("\n[阶段6] 写报告...")
    write_synthetic_report(df, model_metric_df, aux)
    write_threshold_report(base_metric, threshold_df, loss_df)
    write_regularization_report(reg_df)
    write_summary_report()

    print("\n全部任务完成！")
    print("生成文件如下：")
    print(f"1. {SYNTHETIC_DATA_PATH}")
    print(f"2. {SYNTHETIC_REPORT_PATH}")
    print(f"3. {THRESHOLD_REPORT_PATH}")
    print(f"4. {REGULARIZATION_REPORT_PATH}")
    print(f"5. {SUMMARY_REPORT_PATH}")
    print(f"6. {FIG_LINEAR_LOGISTIC}")
    print(f"7. {FIG_LOSS_CURVES}")
    print(f"8. {FIG_THRESHOLD}")
    print(f"9. {FIG_REG_METRICS}")
    print(f"10. {FIG_REG_COMPLEXITY}")


if __name__ == "__main__":
    main()