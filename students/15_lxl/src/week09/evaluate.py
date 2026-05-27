"""模块: week09.evaluate
用途: 多重共线性诊断 (VIF) 与 5 折交叉验证评估。
      读取 data_prep.py 生成的清洗后数据，进行模型诊断与基线验证。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# 将 src/ 加入搜索路径，以便导入 utils 和 models
sys.path.append(str(Path(__file__).parent.parent))
from utils.models import AnalyticalOLS
from utils.diagnostics import calculate_vif

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
# STUDENT_ROOT = students/15_lxl/
STUDENT_ROOT = Path(__file__).resolve().parent.parent.parent
# PROJECT_ROOT = 项目根目录 (Regression-Analysis-2026/)
PROJECT_ROOT = STUDENT_ROOT.parent.parent
# 清洗后的数据放在自己的 data/ 目录下
CLEAN_DATA_PATH = STUDENT_ROOT / "data" / "clean_marketing.csv"
# 报告输出目录
RESULTS_DIR = STUDENT_ROOT / "results"


def red_warning(msg: str) -> str:
    """将消息包裹为红色 ANSI 转义序列，用于终端高亮警告输出。"""
    return f"\033[91m⚠ {msg}\033[0m"


def task_vif_diagnosis(X: np.ndarray, feature_names: list):
    """Task 1: 多重共线性诊断。

    调用 calculate_vif 计算每个特征的 VIF 值并打印表格。
    如果某个特征 VIF > 10，用红色字体输出警告，提示存在严重共线性。
    """
    print("=" * 60)
    print("Task 1: 多重共线性诊断 (VIF)")
    print("=" * 60)

    # 调用诊断工具箱中的 VIF 计算函数
    vif_values = calculate_vif(X)

    # 打印 VIF 表格
    print(f"\n  {'特征':<25} {'VIF':>10}")
    print("  " + "-" * 37)
    for name, vif in zip(feature_names, vif_values):
        print(f"  {name:<25} {vif:>10.4f}")

    # 筛选 VIF > 10 的特征，输出红色警告
    high_vif = [(name, vif) for name, vif in zip(feature_names, vif_values) if vif > 10]
    if high_vif:
        print()
        print(red_warning("检测到严重多重共线性 (VIF > 10):"))
        for name, vif in high_vif:
            print(red_warning(f"  - {name}: VIF = {vif:.4f}"))
        print(red_warning("建议: 考虑移除或合并高 VIF 特征，以改善模型稳定性。"))
    else:
        print("\n  所有特征 VIF < 10，未检测到严重多重共线性。")

    return vif_values


def task_baseline_cv(X: np.ndarray, y: np.ndarray):
    """Task 2: 5 折交叉验证基线评估。

    使用 KFold(n_splits=5) 将数据分为 5 折，
    每次用 4 折训练、1 折验证，计算 R² 并求平均。
    """
    print("\n" + "=" * 60)
    print("Task 2: 5 折交叉验证 — AnalyticalOLS")
    print("=" * 60)

    # 创建 5 折交叉验证器，shuffle=True 保证随机打乱
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []

    # 逐折训练和评估
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        # 按索引划分训练集和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 用训练集拟合 OLS 模型
        model = AnalyticalOLS().fit(X_train, y_train)
        # 在验证集上预测
        preds = model.predict(X_val)

        # 计算该折的 R² 分数
        fold_r2 = r2_score(y_val, preds)
        r2_scores.append(fold_r2)
        print(f"  Fold {fold}: R² = {fold_r2:.4f}")

    # 输出平均 R² 和标准差
    avg_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    print(f"\n  平均 R²: {avg_r2:.4f} (±{std_r2:.4f})")
    return r2_scores


def main():
    # 确保报告输出目录存在
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- 读取清洗后的数据 ----
    print(f"读取清洗后数据: {CLEAN_DATA_PATH}")
    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"  形状: {df.shape}")
    print(f"  列: {list(df.columns)}")

    # ---- 分离特征矩阵 X 和目标向量 y ----
    target_col = "Sales"
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    print(f"\n  特征数: {len(feature_cols)}, 样本数: {len(y)}")

    # ---- Task 1: VIF 诊断 ----
    vif_values = task_vif_diagnosis(X, feature_cols)

    # ---- Task 2: 基线 5 折交叉验证 ----
    r2_scores = task_baseline_cv(X, y)

    # ---- 写入中文总结报告 ----
    report_lines = [
        "# 第九周 — 总结报告",
        "",
        "## 1. 数据清洗",
        "",
        "- 使用 `data_prep.py` CLI 脚本对 `dirty_marketing.csv` 进行清洗。",
        "- **缺失值处理**: TV_Budget 列有 50 个缺失值，使用全局均值填充。",
        "- **异常值处理**: 对预算列进行 99 分位数缩尾 (Winsorization)。",
        "- **分类变量编码**: 对 Region 列进行 One-Hot 编码 (drop_first=True 避免虚拟变量陷阱)。",
        "",
        "## 2. 多重共线性诊断 (VIF)",
        "",
        "| 特征 | VIF |",
        "|---|---|",
    ]
    # 填充 VIF 表格行
    for name, vif in zip(feature_cols, vif_values):
        flag = " ⚠️" if vif > 10 else ""
        report_lines.append(f"| {name} | {vif:.4f}{flag} |")

    # 根据是否有高 VIF 特征，添加警告或正常说明
    high_vif = [(n, v) for n, v in zip(feature_cols, vif_values) if v > 10]
    if high_vif:
        report_lines += [
            "",
            "**警告**: 以下特征 VIF > 10，存在严重多重共线性:",
        ]
        for n, v in high_vif:
            report_lines.append(f"- {n}: VIF = {v:.4f}")
    else:
        report_lines += ["", "所有特征 VIF < 10，未检测到严重多重共线性。"]

    # 添加交叉验证结果
    report_lines += [
        "",
        "## 3. 5 折交叉验证基线 (AnalyticalOLS)",
        "",
        "| Fold | R² |",
        "|---|---|",
    ]
    for i, r2 in enumerate(r2_scores, start=1):
        report_lines.append(f"| {i} | {r2:.4f} |")

    # 添加课堂讨论思考
    report_lines += [
        "",
        f"**平均 R²**: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})",
        "",
        "## 4. 课堂讨论思考",
        "",
        "在 `data_prep.py` 中，我们使用**全量数据的均值**来填充缺失值。",
        "这意味着在 5 折交叉验证中，验证集的缺失值已经被全量数据的均值'填充'过了，",
        "验证集并非完全未见过的'陌生数据'。这会导致交叉验证的 R² 偏高，",
        "产生过于乐观的泛化估计。正确的做法应该是在每一折中，仅用训练集的均值来填充。",
    ]

    # 写入报告文件
    report_path = RESULTS_DIR / "week09_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n报告已保存 → {report_path}")


if __name__ == "__main__":
    main()
