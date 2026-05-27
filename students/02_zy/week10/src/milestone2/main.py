import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1]
WEEK10_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(SRC_DIR))

from utils.metrics import calculate_mae, calculate_mape, calculate_rmse
from utils.models import GradientDescentOLS
from utils.transformers import CustomStandardScaler


RANDOM_SEED = 42
N_FOLDS = 5


def find_data_path() -> Path:
    """
    Use the data file under week10/data.

    No absolute path is used.
    """
    data_path = WEEK10_DIR / "data" / "dirty_marketing.csv"

    if not data_path.exists():
        print("Error: data file was not found.")
        print("Please put the data file here:")
        print("week10/data/dirty_marketing.csv")
        raise SystemExit(1)

    return data_path


def prepare_results_dir() -> Path:
    """
    Clean and recreate the results directory every time the program runs.
    """
    results_dir = WEEK10_DIR / "results"

    if results_dir.exists():
        shutil.rmtree(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir


def load_dataset(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def choose_target_column(df: pd.DataFrame) -> str:
    """
    Automatically choose the target column.

    Priority:
    Sales -> sales -> Revenue -> revenue -> target -> last numeric column
    """
    candidates = [
        "Sales",
        "sales",
        "Revenue",
        "revenue",
        "Profit",
        "profit",
        "target",
        "Target",
        "y",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) == 0:
        print("Error: no numeric target column found.")
        raise SystemExit(1)

    return numeric_cols[-1]


def build_xy(df: pd.DataFrame, target_col: str):
    """
    Split dataframe into X and y.

    Categorical variables are converted into dummy variables.
    drop_first=True is used to avoid the dummy variable trap.
    """
    df = df.copy()

    # Target cannot be missing.
    df = df.dropna(subset=[target_col])

    y = df[target_col].astype(float).to_numpy()

    X_df = df.drop(columns=[target_col])

    X_df = pd.get_dummies(X_df, drop_first=True)

    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    feature_names = X_df.columns.tolist()

    X = X_df.to_numpy(dtype=float)

    return X, y, feature_names


def make_folds(n_samples: int, n_folds: int = N_FOLDS):
    indices = np.arange(n_samples)

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(indices)

    return np.array_split(indices, n_folds)


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
    }


def average_metrics(fold_metrics: list[dict]) -> dict:
    return {
        "RMSE": float(np.mean([m["RMSE"] for m in fold_metrics])),
        "MAE": float(np.mean([m["MAE"] for m in fold_metrics])),
        "MAPE": float(np.mean([m["MAPE"] for m in fold_metrics])),
    }


def bad_cross_validation(df: pd.DataFrame, target_col: str):
    """
    Task 3: Bad Cross Validation.

    This function intentionally creates data leakage.

    Wrong process:
    1. Use all X to fit CustomStandardScaler.
    2. Fill all missing values using global means.
    3. Standardize the whole dataset before cross-validation.
    4. Then run 5-fold CV.

    Since validation data participates in preprocessing,
    the validation set is no longer truly unseen.
    """
    X, y, _ = build_xy(df, target_col)

    scaler = CustomStandardScaler()

    # Global preprocessing: this is intentionally wrong.
    X_scaled = scaler.fit_transform(X)

    folds = make_folds(len(y))

    fold_metrics = []

    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)

        X_train = X_scaled[train_idx]
        X_val = X_scaled[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        model = GradientDescentOLS(
            learning_rate=0.01,
            n_iterations=10000,
            tolerance=1e-8,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        metrics = evaluate_predictions(y_val, y_pred)
        metrics["Fold"] = fold_id

        fold_metrics.append(metrics)

    return average_metrics(fold_metrics), fold_metrics


def good_cross_validation(df: pd.DataFrame, target_col: str):
    """
    Task 4: Good Cross Validation.

    This function avoids data leakage.

    Correct process inside each fold:
    1. Split X_train and X_val first.
    2. Fit CustomStandardScaler only on X_train.
    3. Transform X_train using training statistics.
    4. Transform X_val using the same training statistics.
    5. Train model on X_train and evaluate on X_val.

    Important:
    X_val is never used in scaler.fit().
    """
    X, y, _ = build_xy(df, target_col)

    folds = make_folds(len(y))

    fold_metrics = []

    for fold_id, val_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)

        X_train_raw = X[train_idx]
        X_val_raw = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        scaler = CustomStandardScaler()

        # Fit only on training data.
        X_train = scaler.fit_transform(X_train_raw)

        # Validation data can only be transformed.
        # Do not call fit() or fit_transform() on X_val.
        X_val = scaler.transform(X_val_raw)
        

        model = GradientDescentOLS(
            learning_rate=0.01,
            n_iterations=10000,
            tolerance=1e-8,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        metrics = evaluate_predictions(y_val, y_pred)
        metrics["Fold"] = fold_id

        fold_metrics.append(metrics)

    return average_metrics(fold_metrics), fold_metrics


def make_summary_table(bad_summary: dict, good_summary: dict) -> str:
    lines = [
        "| Method | RMSE | MAE | MAPE (%) |",
        "|---|---:|---:|---:|",
        (
            "| Task 3 Bad CV: Global preprocessing leakage | "
            f"{bad_summary['RMSE']:.4f} | "
            f"{bad_summary['MAE']:.4f} | "
            f"{bad_summary['MAPE']:.4f} |"
        ),
        (
            "| Task 4 Good CV: Leakage-free pipeline | "
            f"{good_summary['RMSE']:.4f} | "
            f"{good_summary['MAE']:.4f} | "
            f"{good_summary['MAPE']:.4f} |"
        ),
    ]

    return "\n".join(lines)


def make_fold_table(title: str, fold_metrics: list[dict]) -> str:
    lines = [
        f"### {title}",
        "",
        "| Fold | RMSE | MAE | MAPE (%) |",
        "|---:|---:|---:|---:|",
    ]

    for metrics in fold_metrics:
        lines.append(
            f"| {metrics['Fold']} | "
            f"{metrics['RMSE']:.4f} | "
            f"{metrics['MAE']:.4f} | "
            f"{metrics['MAPE']:.4f} |"
        )

    return "\n".join(lines)


def write_report(
    results_dir: Path,
    target_col: str,
    bad_summary: dict,
    good_summary: dict,
    bad_folds: list[dict],
    good_folds: list[dict],
) -> None:
    report_path = results_dir / "evaluation_comparison.md"

    content = "\n".join(
        [
            "# Milestone Project 2：工业流水线与无泄漏泛化评估",
            "",
            "## 一、目标变量",
            "",
            f"本次实验使用的目标变量是：`{target_col}`。",
            "",
            "## 二、平均交叉验证指标对比",
            "",
            make_summary_table(bad_summary, good_summary),
            "",
            "## 三、每一折交叉验证结果",
            "",
            make_fold_table("Task 3：存在数据泄露的交叉验证", bad_folds),
            "",
            make_fold_table("Task 4：无数据泄露的交叉验证", good_folds),
            "",
            "## 四、为什么 Task 3 是危险的？",
            "",
            "Task 3 在交叉验证之前就对全量数据进行了预处理。也就是说，"
            "标准化器在计算均值和标准差时，使用了整个数据集的信息，"
            "其中既包括训练集，也包括验证集。缺失值填补同样使用了全局均值。",
            "",
            "这样做会造成数据泄露，因为验证集的信息提前参与了数据清洗和标准化过程。"
            "因此，验证集不再是真正意义上的未知数据。模型在这种情况下得到的评估结果"
            "可能会显得更好，但这种好成绩是不可靠的。",
            "",
            "## 五、为什么 Task 4 是无泄漏的？",
            "",
            "Task 4 将预处理过程放在了每一折交叉验证的循环内部。"
            "在每一折中，程序先划分 `X_train` 和 `X_val`，然后只使用 `X_train` "
            "去拟合 `CustomStandardScaler`，得到训练集的均值和标准差。",
            "",
            "随后，程序使用训练集学到的均值和标准差去转换 `X_train` 和 `X_val`。"
            "整个过程中，验证集 `X_val` 从未参与 `.fit()` 操作，因此避免了数据泄露。",
            "",
            "这种流程更接近真实业务场景：模型上线之后，未来数据是不可能提前参与训练阶段"
            "的数据清洗和标准化过程的。",
            "",
            "## 六、业务解释",
            "",
            "从业务角度来看，MAE 和 MAPE 比 R² 更容易解释。",
            "",
            "MAE 表示模型预测值与真实值之间的平均绝对误差，可以理解为模型平均会预测错多少销售额。"
            "MAPE 表示平均百分比误差，可以理解为模型平均会有百分之多少的预测偏差。",
            "",
            "在向业务团队或管理者汇报时，应该优先参考 Task 4 的结果，而不是 Task 3 的结果。"
            "虽然 Task 3 的结果可能看起来更好，但它包含了数据泄露，不能代表模型上线后的真实表现。"
            "Task 4 的无泄漏结果才更接近真实部署环境下的泛化误差。",
            "",
        ]
    )

    report_path.write_text(content, encoding="utf-8")


def create_optional_plot(results_dir: Path, bad_summary: dict, good_summary: dict) -> None:
    """
    Optional artifact:
    Create a bar chart comparing Task 3 and Task 4 metrics.
    """
    try:
        import matplotlib.pyplot as plt

        metric_names = ["RMSE", "MAE", "MAPE"]

        bad_values = [bad_summary[name] for name in metric_names]
        good_values = [good_summary[name] for name in metric_names]

        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(x - width / 2, bad_values, width, label="Bad CV")
        ax.bar(x + width / 2, good_values, width, label="Good CV")

        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_ylabel("Metric Value")
        ax.set_title("Leakage vs Leakage-Free Cross Validation")
        ax.legend()

        fig.tight_layout()

        output_path = results_dir / "leakage_analysis.png"
        fig.savefig(output_path, dpi=200)

        plt.close(fig)

    except Exception as exc:
        print(f"Optional plot was not created: {exc}")


def main() -> None:
    results_dir = prepare_results_dir()

    data_path = find_data_path()

    print("Using data file:", data_path)

    df = load_dataset(data_path)

    target_col = choose_target_column(df)

    print("Target column:", target_col)

    bad_summary, bad_folds = bad_cross_validation(df, target_col)

    good_summary, good_folds = good_cross_validation(df, target_col)

    write_report(
        results_dir=results_dir,
        target_col=target_col,
        bad_summary=bad_summary,
        good_summary=good_summary,
        bad_folds=bad_folds,
        good_folds=good_folds,
    )

    create_optional_plot(results_dir, bad_summary, good_summary)

    print()
    print("===== Task 3: Bad Cross Validation =====")
    print(f"Average RMSE: {bad_summary['RMSE']:.4f}")
    print(f"Average MAE : {bad_summary['MAE']:.4f}")
    print(f"Average MAPE: {bad_summary['MAPE']:.4f}%")

    print()
    print("===== Task 4: Good Cross Validation =====")
    print(f"Average RMSE: {good_summary['RMSE']:.4f}")
    print(f"Average MAE : {good_summary['MAE']:.4f}")
    print(f"Average MAPE: {good_summary['MAPE']:.4f}%")

    print()
    print("Results saved to:")
    print(results_dir / "evaluation_comparison.md")
    print(results_dir / "leakage_analysis.png")


if __name__ == "__main__":
    main()