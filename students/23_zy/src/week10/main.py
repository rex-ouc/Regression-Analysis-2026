"""
Week10 / Milestone Project 2

主题：
1. 实现有数据泄露的交叉验证 bad_cross_validation
2. 实现无数据泄露的交叉验证 good_cross_validation
3. 比较 RMSE、MAE、MAPE
4. 自动生成 results/evaluation_comparison.md
5. 自动生成 results/leakage_analysis.png

运行方式：
    uv run src/milestone2/main.py
"""

import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 解决导入路径问题
# ------------------------------------------------------------
# 当前文件位置是：
# students/23_zy/src/milestone2/main.py
#
# 但是 utils 在：
# students/23_zy/src/utils/
#
# 所以这里把 src 目录加入 Python 搜索路径，方便导入 utils。
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


from utils.metrics import calculate_mae, calculate_mape, calculate_rmse
from utils.models import GradientDescentOLS
from utils.transformers import CustomStandardScaler


# ------------------------------------------------------------
# 路径设置
# ------------------------------------------------------------
DATA_PATH = os.path.join(PROJECT_DIR, "data", "dirty_q4_marketing.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


# ------------------------------------------------------------
# 工具函数 1：准备 results 文件夹
# ------------------------------------------------------------
def prepare_results_dir():
    """
    每次运行程序前，都重新创建 results 文件夹。

    这样可以保证：
    1. 旧结果不会影响新结果
    2. 作业输出文件位置固定
    """
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)

    os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------
# 工具函数 2：读取数据
# ------------------------------------------------------------
def load_data():
    """
    读取 dirty_q4_marketing.csv。

    这里不使用 C:/Users/... 这种绝对路径，
    而是使用相对路径自动拼接，避免老师运行时报 FileNotFoundError。
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"没有找到数据文件：{DATA_PATH}\n"
            "请检查 dirty_q4_marketing.csv 是否放在 students/23_zy/data/ 目录下。"
        )

    data = pd.read_csv(DATA_PATH)

    print("成功读取数据：")
    print(f"数据路径：{DATA_PATH}")
    print(f"数据形状：{data.shape}")
    print("数据列名：", list(data.columns))

    return data


# ------------------------------------------------------------
# 工具函数 3：拆分特征 X 和目标 y
# ------------------------------------------------------------
def split_features_target(data):
    """
    自动寻找目标变量 y。

    一般营销数据里目标变量可能叫：
    Sales, sales, Revenue, revenue, target, y

    如果找不到，就默认最后一列是目标变量。
    """

    possible_targets = ["Sales", "sales", "Revenue", "revenue", "target", "Target", "y"]

    target_col = None
    for col in possible_targets:
        if col in data.columns:
            target_col = col
            break

    if target_col is None:
        target_col = data.columns[-1]
        print(f"没有找到常见目标列名，默认使用最后一列作为 y：{target_col}")
    else:
        print(f"识别到目标变量 y：{target_col}")

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # 只保留数值型特征，避免字符串列导致模型报错
    X = X.select_dtypes(include=[np.number])

    # y 转成数值，如果里面有异常字符，无法转换的会变成 NaN
    y = pd.to_numeric(y, errors="coerce")

    # 删除 y 缺失的样本
    valid_mask = ~y.isna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    print(f"用于建模的特征列：{list(X.columns)}")
    print(f"有效样本数量：{len(y)}")

    return X.to_numpy(dtype=float), y.to_numpy(dtype=float), list(X.columns), target_col


# ------------------------------------------------------------
# 工具函数 4：手写 K 折切分
# ------------------------------------------------------------
def make_kfold_indices(n_samples, n_splits=5, random_state=42):
    """
    手写 5 折交叉验证的索引切分。

    不用 sklearn，避免依赖太复杂。
    """
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    folds = []
    current = 0

    for fold_size in fold_sizes:
        start = current
        stop = current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, val_indices))
        current = stop

    return folds


# ------------------------------------------------------------
# Task 3：有数据泄露的交叉验证
# ------------------------------------------------------------
def bad_cross_validation(X, y, n_splits=5):
    """
    错误示范：有数据泄露的交叉验证。

    错在哪里？
    1. 先对全体 X 计算均值并填补缺失值
    2. 再对全体 X 做标准化 fit_transform
    3. 然后才做 5 折交叉验证

    这样验证集的信息提前参与了预处理，所以叫数据泄露。
    """

    print("\n========== Task 3：bad_cross_validation，有数据泄露 ==========")

    # 1. 全局均值填补缺失值
    global_mean = np.nanmean(X, axis=0)
    X_filled = np.where(np.isnan(X), global_mean, X)

    # 2. 对全体数据 fit_transform
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    folds = make_kfold_indices(len(y), n_splits=n_splits, random_state=42)

    rmse_list = []
    mae_list = []
    mape_list = []

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        X_train = X_scaled[train_idx]
        X_val = X_scaled[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        model = GradientDescentOLS(
            learning_rate=0.01,
            n_epochs=3000,
            random_state=42,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = calculate_rmse(y_val, y_pred)
        mae = calculate_mae(y_val, y_pred)
        mape = calculate_mape(y_val, y_pred)

        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)

        print(
            f"Fold {fold_id}: "
            f"RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}%"
        )

    result = {
        "method": "Bad CV with Data Leakage",
        "rmse": float(np.mean(rmse_list)),
        "mae": float(np.mean(mae_list)),
        "mape": float(np.mean(mape_list)),
    }

    print(
        f"平均结果：RMSE={result['rmse']:.4f}, "
        f"MAE={result['mae']:.4f}, MAPE={result['mape']:.4f}%"
    )

    return result


# ------------------------------------------------------------
# Task 4：无数据泄露的交叉验证
# ------------------------------------------------------------
def good_cross_validation(X, y, n_splits=5):
    """
    正确示范：无数据泄露的交叉验证。

    核心原则：
    每一折里面，只能用训练集学习预处理参数。

    具体做法：
    1. 先切分 X_train 和 X_val
    2. 用 X_train 的均值填补 X_train 和 X_val
    3. scaler 只能 fit(X_train)
    4. X_val 只能 transform，绝对不能 fit
    """

    print("\n========== Task 4：good_cross_validation，无数据泄露 ==========")

    folds = make_kfold_indices(len(y), n_splits=n_splits, random_state=42)

    rmse_list = []
    mae_list = []
    mape_list = []

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        X_train = X[train_idx].copy()
        X_val = X[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]

        # 1. 只用训练集计算均值
        train_mean = np.nanmean(X_train, axis=0)

        # 如果某列全是 NaN，nanmean 会得到 NaN，这里把它替换成 0
        train_mean = np.where(np.isnan(train_mean), 0.0, train_mean)

        # 2. 用训练集均值填补训练集和验证集
        X_train_filled = np.where(np.isnan(X_train), train_mean, X_train)
        X_val_filled = np.where(np.isnan(X_val), train_mean, X_val)

        # 3. scaler 只在训练集上 fit
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)

        # 4. 验证集只能 transform，不能 fit
        X_val_scaled = scaler.transform(X_val_filled)

        model = GradientDescentOLS(
            learning_rate=0.01,
            n_epochs=3000,
            random_state=42,
        )

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        rmse = calculate_rmse(y_val, y_pred)
        mae = calculate_mae(y_val, y_pred)
        mape = calculate_mape(y_val, y_pred)

        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)

        print(
            f"Fold {fold_id}: "
            f"RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}%"
        )

    result = {
        "method": "Good CV without Data Leakage",
        "rmse": float(np.mean(rmse_list)),
        "mae": float(np.mean(mae_list)),
        "mape": float(np.mean(mape_list)),
    }

    print(
        f"平均结果：RMSE={result['rmse']:.4f}, "
        f"MAE={result['mae']:.4f}, MAPE={result['mape']:.4f}%"
    )

    return result


# ------------------------------------------------------------
# Task 5：保存 Markdown 报告
# ------------------------------------------------------------
def save_markdown_report(bad_result, good_result, feature_names, target_col):
    """
    保存作业要求的 evaluation_comparison.md。
    """
    output_path = os.path.join(RESULTS_DIR, "evaluation_comparison.md")

    content = f"""# Week10 Milestone Project 2：Evaluation Comparison

## 1. Dataset Information

- Target variable: `{target_col}`
- Feature variables: {", ".join([f"`{name}`" for name in feature_names])}
- Cross validation: 5-Fold CV
- Model: GradientDescentOLS

## 2. Metrics Comparison

| Method | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Bad CV with Data Leakage | {bad_result["rmse"]:.4f} | {bad_result["mae"]:.4f} | {bad_result["mape"]:.4f}% |
| Good CV without Data Leakage | {good_result["rmse"]:.4f} | {good_result["mae"]:.4f} | {good_result["mape"]:.4f}% |

## 3. Analysis

本次实验主要比较了两种交叉验证方式：一种是存在数据泄露的 bad_cross_validation，另一种是更加严格的 good_cross_validation。

在 bad_cross_validation 中，程序先对全体数据进行了缺失值填补和标准化，然后才进行 5 折交叉验证。这样做表面上很方便，但是验证集的信息已经提前参与了数据预处理过程，所以模型评估结果会偏乐观。

在 good_cross_validation 中，每一折都会先划分训练集和验证集。缺失值填补的均值、标准化的均值和标准差，都只从训练集中学习得到。验证集只使用训练集得到的参数进行 transform，不重新 fit。因此，这种方式更接近真实业务上线后的情况。

虽然 good_cross_validation 的误差可能会比 bad_cross_validation 更大，看起来成绩没有那么好，但它更加可信。因为真实预测时，未来数据是不可能提前参与模型训练和预处理的。如果为了让指标好看而使用数据泄露的结果，模型上线后很可能表现不稳定，最终会误导业务判断。

## 4. Business Explanation

从业务角度看，MAE 可以理解为模型平均预测会偏差多少销售额或预算金额，MAPE 可以理解为平均百分比误差。

因此，在向业务团队汇报时，更应该使用 good_cross_validation 的结果。它虽然可能更保守，但是能更真实地反映模型面对新数据时的预测能力。
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nMarkdown 报告已保存：{output_path}")


# ------------------------------------------------------------
# Task 5：保存柱状图
# ------------------------------------------------------------
def save_bar_plot(bad_result, good_result):
    """
    保存 leakage_analysis.png。

    图中比较两种方法下的 RMSE、MAE、MAPE。
    """
    output_path = os.path.join(RESULTS_DIR, "leakage_analysis.png")

    metrics = ["RMSE", "MAE", "MAPE"]
    bad_values = [bad_result["rmse"], bad_result["mae"], bad_result["mape"]]
    good_values = [good_result["rmse"], good_result["mae"], good_result["mape"]]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, bad_values, width, label="Bad CV with Leakage")
    plt.bar(x + width / 2, good_values, width, label="Good CV without Leakage")

    plt.xticks(x, metrics)
    plt.ylabel("Error")
    plt.title("Error Comparison: Data Leakage vs Leakage-Free CV")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"柱状图已保存：{output_path}")


# ------------------------------------------------------------
# 主函数
# ------------------------------------------------------------
def main():
    """
    程序唯一入口。

    在终端中运行：
        uv run src/milestone2/main.py
    """
    print("========== Week10 Milestone Project 2 开始运行 ==========")

    prepare_results_dir()

    data = load_data()
    X, y, feature_names, target_col = split_features_target(data)

    bad_result = bad_cross_validation(X, y, n_splits=5)
    good_result = good_cross_validation(X, y, n_splits=5)

    save_markdown_report(bad_result, good_result, feature_names, target_col)
    save_bar_plot(bad_result, good_result)

    print("\n========== 全部任务完成 ==========")
    print("请检查 results 文件夹，里面应该有：")
    print("1. evaluation_comparison.md")
    print("2. leakage_analysis.png")


if __name__ == "__main__":
    main()