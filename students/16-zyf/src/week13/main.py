import os
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from src.utils.metrics import calculate_rmse, calculate_mae
from src.utils.transformers import CustomStandardScaler


warnings.filterwarnings("ignore")


DATA_DIR = "src/week13/data"
RESULTS_DIR = "src/week13/results"
FIGURES_DIR = "src/week13/results/figures"

SYNTHETIC_PATH = os.path.join(DATA_DIR, "synthetic_correlated.csv")
KAGGLE_WEEK11_PATH = "src/week11/data/train.csv"
KAGGLE_WEEK13_PATH = os.path.join(DATA_DIR, "kaggle_train.csv")

TARGET_COL = "TARGET(PRICE_IN_LACS)"


# =========================================================
# 让自己写的 CustomStandardScaler 可以放进 sklearn Pipeline
# =========================================================
class PipelineCustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = CustomStandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)


# =========================================================
# 基础工具
# =========================================================
def prepare_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)

    os.makedirs(FIGURES_DIR, exist_ok=True)


def dataframe_to_markdown(df, digits=4):
    df = df.copy()
    df = df.round(digits)

    headers = list(df.columns)
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def evaluate_model(y_true, y_pred):
    return {
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred),
    }


# =========================================================
# Task A1：生成高度共线性模拟数据
# =========================================================
def generate_synthetic_correlated_data(n_samples=500, random_state=42):
    rng = np.random.default_rng(random_state)

    base_marketing = rng.normal(0, 1, n_samples)

    x1 = base_marketing + rng.normal(0, 0.05, n_samples)
    x2 = base_marketing + rng.normal(0, 0.05, n_samples)
    x3 = base_marketing + rng.normal(0, 0.05, n_samples)

    x4 = rng.normal(0, 1, n_samples)
    x5 = rng.normal(0, 1, n_samples)

    noise1 = rng.normal(0, 1, n_samples)
    noise2 = rng.normal(0, 1, n_samples)
    noise3 = rng.normal(0, 1, n_samples)

    y = (
        5
        + 3.0 * x1
        - 2.0 * x4
        + 1.5 * x5
        + rng.normal(0, 1.0, n_samples)
    )

    df = pd.DataFrame(
        {
            "x1_marketing": x1,
            "x2_marketing_copy": x2,
            "x3_marketing_copy": x3,
            "x4_price": x4,
            "x5_service": x5,
            "noise1": noise1,
            "noise2": noise2,
            "noise3": noise3,
            "y": y,
        }
    )

    df.to_csv(SYNTHETIC_PATH, index=False)
    return df


# =========================================================
# OLS vs Ridge：50 次随机切分，比较系数稳定性
# =========================================================
def coefficient_stability_experiment(df, n_repeats=50):
    feature_cols = [c for c in df.columns if c != "y"]
    correlated_features = [
        "x1_marketing",
        "x2_marketing_copy",
        "x3_marketing_copy",
    ]

    rows = []

    X = df[feature_cols].values
    y = df["y"].values

    for seed in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=seed,
        )

        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_train_scaled, y_train)

        for feature in correlated_features:
            idx = feature_cols.index(feature)

            rows.append(
                {
                    "model": "OLS",
                    "feature": feature,
                    "coef": ols.coef_[idx],
                }
            )

            rows.append(
                {
                    "model": "Ridge_alpha_10",
                    "feature": feature,
                    "coef": ridge.coef_[idx],
                }
            )

    coef_df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))
    labels = []
    data = []

    for model in ["OLS", "Ridge_alpha_10"]:
        for feature in correlated_features:
            values = coef_df[
                (coef_df["model"] == model)
                & (coef_df["feature"] == feature)
            ]["coef"].values

            labels.append(f"{model}\n{feature}")
            data.append(values)

    plt.boxplot(data, labels=labels)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Coefficient")
    plt.title("OLS vs Ridge Coefficient Stability")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "synthetic_ols_vs_ridge_boxplot.png")
    plt.savefig(path, dpi=300)
    plt.close()

    stability_summary = (
        coef_df.groupby(["model", "feature"])["coef"]
        .std()
        .reset_index()
        .rename(columns={"coef": "coef_std"})
    )

    return coef_df, stability_summary


# =========================================================
# GridSearchCV：Ridge / Lasso / ElasticNet 调参
# =========================================================
def tune_regularized_models(X_train, y_train):
    alphas = np.logspace(-4, 3, 50)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    ridge_pipe = Pipeline(
        [
            ("scaler", PipelineCustomScaler()),
            ("model", Ridge()),
        ]
    )

    lasso_pipe = Pipeline(
        [
            ("scaler", PipelineCustomScaler()),
            ("model", Lasso(max_iter=20000)),
        ]
    )

    elastic_pipe = Pipeline(
        [
            ("scaler", PipelineCustomScaler()),
            ("model", ElasticNet(max_iter=20000)),
        ]
    )

    ridge_grid = GridSearchCV(
        ridge_pipe,
        {"model__alpha": alphas},
        cv=cv,
        scoring="neg_root_mean_squared_error",
    )

    lasso_grid = GridSearchCV(
        lasso_pipe,
        {"model__alpha": alphas},
        cv=cv,
        scoring="neg_root_mean_squared_error",
    )

    elastic_grid = GridSearchCV(
        elastic_pipe,
        {
            "model__alpha": alphas,
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        cv=cv,
        scoring="neg_root_mean_squared_error",
    )

    ridge_grid.fit(X_train, y_train)
    lasso_grid.fit(X_train, y_train)
    elastic_grid.fit(X_train, y_train)

    return ridge_grid, lasso_grid, elastic_grid


def plot_cv_curve(grid, title, filename):
    results = pd.DataFrame(grid.cv_results_)

    if "param_model__l1_ratio" in results.columns:
        best_l1 = grid.best_params_["model__l1_ratio"]
        results = results[results["param_model__l1_ratio"] == best_l1]

    alphas = results["param_model__alpha"].astype(float)
    rmse = -results["mean_test_score"]

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, rmse, marker="o")
    plt.xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("CV RMSE")
    plt.title(title)

    best_alpha = grid.best_params_["model__alpha"]
    best_rmse = -grid.best_score_

    plt.axvline(best_alpha, linestyle="--")
    plt.text(best_alpha, best_rmse, f"best alpha={best_alpha:.4f}")

    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()


def extract_coefficients(model, feature_names):
    reg = model.best_estimator_.named_steps["model"]

    return pd.DataFrame(
        {
            "feature": feature_names,
            "coef": reg.coef_,
        }
    )


# =========================================================
# 自己实现：前向选择 Top-K
# =========================================================
def forward_selection_top_k(X, y, feature_names, k=5):
    selected = []
    remaining = list(range(X.shape[1]))

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    while len(selected) < k and remaining:
        best_feature = None
        best_score = np.inf

        for feature_idx in remaining:
            candidate = selected + [feature_idx]
            scores = []

            for train_idx, val_idx in cv.split(X):
                X_train = X[train_idx][:, candidate]
                X_val = X[val_idx][:, candidate]

                y_train = y[train_idx]
                y_val = y[val_idx]

                scaler = CustomStandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                model = LinearRegression()
                model.fit(X_train_scaled, y_train)

                pred = model.predict(X_val_scaled)
                scores.append(calculate_rmse(y_val, pred))

            avg_score = np.mean(scores)

            if avg_score < best_score:
                best_score = avg_score
                best_feature = feature_idx

        selected.append(best_feature)
        remaining.remove(best_feature)

    selected_features = [feature_names[i] for i in selected]
    return selected_features


# =========================================================
# Task A：完整模拟数据流程
# =========================================================
def run_synthetic_task():
    print("[Task A] Generating synthetic correlated data...")
    df = generate_synthetic_correlated_data()

    feature_cols = [c for c in df.columns if c != "y"]

    X = df[feature_cols].values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    print("[Task A] Running coefficient stability experiment...")
    coef_df, stability_summary = coefficient_stability_experiment(df)

    print("[Task A] Tuning Ridge, Lasso, ElasticNet...")
    ridge_grid, lasso_grid, elastic_grid = tune_regularized_models(
        X_train,
        y_train,
    )

    plot_cv_curve(
        ridge_grid,
        "Ridge CV Curve",
        "synthetic_ridge_cv_curve.png",
    )

    plot_cv_curve(
        lasso_grid,
        "Lasso CV Curve",
        "synthetic_lasso_cv_curve.png",
    )

    plot_cv_curve(
        elastic_grid,
        "ElasticNet CV Curve",
        "synthetic_elasticnet_cv_curve.png",
    )

    models = {
        "OLS": Pipeline(
            [
                ("scaler", PipelineCustomScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": ridge_grid.best_estimator_,
        "Lasso": lasso_grid.best_estimator_,
        "ElasticNet": elastic_grid.best_estimator_,
    }

    eval_rows = []

    for name, model in models.items():
        if name == "OLS":
            model.fit(X_train, y_train)

        pred = model.predict(X_test)
        metrics = evaluate_model(y_test, pred)

        eval_rows.append(
            {
                "model": name,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
            }
        )

    eval_df = pd.DataFrame(eval_rows)

    ridge_coef = extract_coefficients(ridge_grid, feature_cols)
    lasso_coef = extract_coefficients(lasso_grid, feature_cols)
    elastic_coef = extract_coefficients(elastic_grid, feature_cols)

    lasso_selected = (
        lasso_coef[lasso_coef["coef"].abs() > 1e-6]["feature"].tolist()
    )

    forward_selected = forward_selection_top_k(
        X,
        y,
        feature_cols,
        k=5,
    )

    write_synthetic_report(
        eval_df,
        stability_summary,
        ridge_grid,
        lasso_grid,
        elastic_grid,
        ridge_coef,
        lasso_coef,
        elastic_coef,
        lasso_selected,
        forward_selected,
    )

    return eval_df, lasso_selected, forward_selected


def write_synthetic_report(
    eval_df,
    stability_summary,
    ridge_grid,
    lasso_grid,
    elastic_grid,
    ridge_coef,
    lasso_coef,
    elastic_coef,
    lasso_selected,
    forward_selected,
):
    path = os.path.join(RESULTS_DIR, "synthetic_report.md")

    content = f"""
# Week 13 Task A Synthetic Report

## 1. Data Generating Process

本任务自己生成了一份高度共线性的模拟回归数据。

真实 DGP 为：

y = 5 + 3.0 * x1_marketing - 2.0 * x4_price + 1.5 * x5_service + noise

其中：

- x1_marketing 是真实有效变量
- x4_price 是真实有效变量
- x5_service 是真实有效变量
- x2_marketing_copy 和 x3_marketing_copy 是 x1 的高度相关复制变量
- noise1、noise2、noise3 是纯噪声变量

高度相关特征组为：

- x1_marketing
- x2_marketing_copy
- x3_marketing_copy

## 2. OLS vs Ridge 系数稳定性

下面是 50 次随机切分后，共线变量系数标准差：

{dataframe_to_markdown(stability_summary)}

从结果可以看到，OLS 在共线变量上的系数波动更明显，而 Ridge 通过 penalty 收缩系数后，通常更加稳定。

对应箱线图保存在：

src/week13/results/figures/synthetic_ols_vs_ridge_boxplot.png

## 3. 为什么正则化前必须标准化

Ridge、Lasso 和 Elastic Net 都会对系数大小进行惩罚。

如果不同特征的量纲差异很大，例如一个变量范围是 0 到 1，另一个变量范围是 0 到 10000，那么 penalty 对不同变量就不公平。

所以在正则化模型前必须标准化。

## 4. GridSearchCV 最优参数

| Model | Best Params | Best CV RMSE |
|---|---|---:|
| Ridge | {ridge_grid.best_params_} | {-ridge_grid.best_score_:.4f} |
| Lasso | {lasso_grid.best_params_} | {-lasso_grid.best_score_:.4f} |
| ElasticNet | {elastic_grid.best_params_} | {-elastic_grid.best_score_:.4f} |

CV 曲线保存在 figures 文件夹中。

## 5. 测试集表现对比

{dataframe_to_markdown(eval_df)}

## 6. Ridge 系数

{dataframe_to_markdown(ridge_coef)}

## 7. Lasso 系数

{dataframe_to_markdown(lasso_coef)}

## 8. Elastic Net 系数

{dataframe_to_markdown(elastic_coef)}

## 9. Lasso 与前向选择变量名单对比

Lasso 选择出的非零变量：

{lasso_selected}

前向选择 Top-5 选择出的变量：

{forward_selected}

## 10. 解释

Ridge 倾向于保留所有变量，只是把系数整体缩小，因此在共线变量组中通常会保留多个相关变量。

Lasso 更像变量筛选工具，可能只保留共线变量组中的一个，把其他变量压缩为 0。

Elastic Net 是 Ridge 和 Lasso 的折中，既有变量筛选能力，也不会像 Lasso 那样过于激进地只保留一个变量。
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Synthetic report saved to: {path}")


# =========================================================
# Task B：使用 Week11 的 Kaggle 房价 train.csv
# =========================================================
def load_kaggle_data():
    if not os.path.exists(KAGGLE_WEEK11_PATH):
        raise FileNotFoundError(
            f"找不到 Week11 数据，请确认文件存在：{KAGGLE_WEEK11_PATH}"
        )

    df = pd.read_csv(KAGGLE_WEEK11_PATH)
    df.to_csv(KAGGLE_WEEK13_PATH, index=False)

    return df


def preprocess_kaggle(df):
    df = df.copy()

    df = df.drop(columns=["ADDRESS"], errors="ignore")

    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

        low = X[col].quantile(0.01)
        high = X[col].quantile(0.99)
        X[col] = X[col].clip(low, high)

    for col in categorical_cols:
        mode_value = X[col].mode()
        fill_value = mode_value.iloc[0] if len(mode_value) > 0 else "Unknown"
        X[col] = X[col].fillna(fill_value)

    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()

    return X.values.astype(float), y.astype(float), feature_names


def run_kaggle_task():
    print("[Task B] Loading Week11 Kaggle data...")
    df = load_kaggle_data()

    X, y, feature_names = preprocess_kaggle(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    print("[Task B] Tuning regularized models...")
    ridge_grid, lasso_grid, elastic_grid = tune_regularized_models(
        X_train,
        y_train,
    )

    models = {
        "OLS": Pipeline(
            [
                ("scaler", PipelineCustomScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": ridge_grid.best_estimator_,
        "Lasso": lasso_grid.best_estimator_,
        "ElasticNet": elastic_grid.best_estimator_,
    }

    eval_rows = []

    for name, model in models.items():
        if name == "OLS":
            model.fit(X_train, y_train)

        pred = model.predict(X_test)
        metrics = evaluate_model(y_test, pred)

        eval_rows.append(
            {
                "model": name,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
            }
        )

    eval_df = pd.DataFrame(eval_rows)

    lasso_coef = extract_coefficients(lasso_grid, feature_names)
    lasso_selected = (
        lasso_coef[lasso_coef["coef"].abs() > 1e-6]
        .sort_values("coef", key=lambda s: s.abs(), ascending=False)
    )

    top5 = lasso_selected.head(5)["feature"].tolist()

    write_kaggle_report(
        df,
        eval_df,
        ridge_grid,
        lasso_grid,
        elastic_grid,
        lasso_selected,
        top5,
    )

    return eval_df, top5


def write_kaggle_report(
    df,
    eval_df,
    ridge_grid,
    lasso_grid,
    elastic_grid,
    lasso_selected,
    top5,
):
    path = os.path.join(RESULTS_DIR, "kaggle_report.md")

    content = f"""
# Week 13 Task B Kaggle Report

## 1. Dataset Information

本任务使用 Week11 的 Kaggle 房价数据 train.csv。

- 样本量：{df.shape[0]}
- 字段数：{df.shape[1]}
- 目标变量：TARGET(PRICE_IN_LACS)
- 每一行代表一套房产记录

这份数据适合练习正则化，因为其中包含面积、房间数、是否在建、是否可入住、经纬度等变量，部分变量之间可能存在相关性。

## 2. Model Performance

{dataframe_to_markdown(eval_df)}

## 3. Best Parameters

| Model | Best Params | Best CV RMSE |
|---|---|---:|
| Ridge | {ridge_grid.best_params_} | {-ridge_grid.best_score_:.4f} |
| Lasso | {lasso_grid.best_params_} | {-lasso_grid.best_score_:.4f} |
| ElasticNet | {elastic_grid.best_params_} | {-elastic_grid.best_score_:.4f} |

## 4. Lasso Selected Features

Lasso 保留下来的变量如下：

{dataframe_to_markdown(lasso_selected)}

## 5. Top 5 Important Factors

如果业务方要求给出最关键的 5 个影响因素，我会参考 Lasso 保留下来的变量以及系数大小。

Top 5 变量为：

{top5}

## 6. Interpretation

如果正则化方法相比 OLS 没有明显提升，可能原因是：

- 数据中特征数量并不算特别高
- 有些变量本身已经是强预测变量
- 线性模型对房价这种复杂问题表达能力有限
- 房价中存在极端值和地段等隐藏因素

Lasso 剔除变量不一定代表这些变量完全没用，只能说明在当前线性模型和当前数据下，它们对预测贡献较弱。
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Kaggle report saved to: {path}")


# =========================================================
# Task C：总结报告
# =========================================================
def write_summary_report(
    synthetic_eval,
    synthetic_lasso_selected,
    synthetic_forward_selected,
    kaggle_eval,
    kaggle_top5,
):
    path = os.path.join(RESULTS_DIR, "summary_comparison.md")

    content = f"""
# Week 13 Summary Comparison Report

## 1. Synthetic Data Results

{dataframe_to_markdown(synthetic_eval)}

模拟数据中，DGP 是已知的，所以我们可以直接判断模型是否找到了真正有效的变量。

Lasso 选择变量：

{synthetic_lasso_selected}

Forward Selection 选择变量：

{synthetic_forward_selected}

## 2. Kaggle Data Results

{dataframe_to_markdown(kaggle_eval)}

Kaggle 房价数据中，真实 DGP 不知道，因此我们只能做推测解释。

Lasso 给出的 Top 5 重要变量：

{kaggle_top5}

## 3. Lasso 在共线变量组中的业务风险

Lasso 面对高度相关变量时，可能只保留其中一个变量，而把其他相关变量压缩为 0。

这在业务上有风险。

因为被压缩为 0 的变量不一定真的没有价值，只可能是它和另一个变量信息重复。

## 4. Elastic Net 如何缓解这个问题

Elastic Net 同时结合 Ridge 和 Lasso。

它既可以筛变量，也可以在相关变量组中保留一部分整体结构。

所以在高度相关变量成组出现时，Elastic Net 往往比 Lasso 更稳健。

## 5. GridSearchCV 与主观偏好

GridSearchCV 的目标是找到验证误差最低的超参数。

但是业务上有时还会追求：

- 模型更稀疏
- 模型更稳定
- 模型更容易解释

所以最低验证误差不一定等于最符合业务解释需求。

## 6. Forward Selection 与 Lasso 对比

Forward Selection 是一步一步加入变量，逻辑直观，但计算成本较高。

Lasso 通过正则化自动实现变量筛选，效率更高。

但是 Lasso 在共线变量组中可能选择不稳定，因此结果解释需要谨慎。

## 7. Final Conclusion

Week13 的核心结论是：

OLS 在共线性场景下系数容易不稳定。

Ridge 适合提升稳定性。

Lasso 适合变量筛选。

Elastic Net 适合处理成组相关变量。

传统变量选择方法可以作为对照，帮助我们理解 Lasso 的筛选结果是否合理。
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Summary report saved to: {path}")


# =========================================================
# 主函数
# =========================================================
def main():
    prepare_dirs()

    synthetic_eval, synthetic_lasso_selected, synthetic_forward_selected = (
        run_synthetic_task()
    )

    kaggle_eval, kaggle_top5 = run_kaggle_task()

    write_summary_report(
        synthetic_eval,
        synthetic_lasso_selected,
        synthetic_forward_selected,
        kaggle_eval,
        kaggle_top5,
    )

    print("\nWeek13 finished.")


if __name__ == "__main__":
    main()