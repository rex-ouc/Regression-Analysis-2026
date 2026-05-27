"""模块: week09.data_prep
用途: 数据清洗命令行脚本 (CLI)。
      支持 --input / --output 参数，处理缺失值、异常值和分类变量。

示例:
    uv run src/week09/data_prep.py \
        --input homework/week09/data/dirty_marketing.csv \
        --output students/15_lxl/data/clean_marketing.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """解析命令行参数。
    --input:  原始脏数据的 CSV 路径（必填）
    --output: 清洗后数据的保存路径（必填）
    """
    parser = argparse.ArgumentParser(description="数据清洗 CLI 脚本")
    parser.add_argument("--input", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output", required=True, help="输出 CSV 文件路径")
    return parser.parse_args()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """处理缺失值: 对每个数值列，用该列的全局均值填充 NaN。

    注意: 这种做法会导致交叉验证时验证集"间接"看到了全量数据的统计信息，
    产生数据泄漏。更严谨的做法是在每折 CV 内部，仅用训练集的均值填充。
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            # 计算该列均值（自动忽略 NaN）
            mean_val = df[col].mean()
            # 用均值填充所有缺失位置
            df[col] = df[col].fillna(mean_val)
            print(f"  [缺失值] {col}: 用均值 {mean_val:.4f} 填充")
    return df


def winsorize_columns(df: pd.DataFrame, columns: list, quantile: float = 0.99) -> pd.DataFrame:
    """缩尾处理 (Winsorization): 将超过 quantile 分位数的极端值截断到该分位数。

    例如 quantile=0.99 表示将大于第 99 百分位数的值全部拉回到第 99 百分位数，
    避免极端异常值对模型产生过大的影响。
    """
    for col in columns:
        # 计算上界（第 99 百分位数）
        upper = df[col].quantile(quantile)
        # 统计有多少个值超过了上界
        n_clipped = (df[col] > upper).sum()
        # 将超过上界的值截断到上界
        df[col] = df[col].clip(upper=upper)
        print(f"  [缩尾] {col}: {quantile*100:.0f}% 分位数 = {upper:.4f}, 截断 {n_clipped} 个极端值")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """对分类变量进行 One-Hot 编码。

    关键: 使用 drop_first=True 丢弃第一列（如 Region_East），
    避免"虚拟变量陷阱"——即所有哑变量加起来恒等于 1，
    导致 X^T X 矩阵奇异（不可逆），OLS 无法求解。
    """
    # 筛选出所有分类类型的列（object / string / category）
    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    if len(categorical_cols) == 0:
        print("  [编码] 无分类变量需要编码")
        return df

    print(f"  [编码] 分类列: {list(categorical_cols)}")
    # drop_first=True: 丢弃第一个类别，防止虚拟变量陷阱
    # dtype=float: 确保生成的是 0.0/1.0 而不是 True/False
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=float)
    print(f"  [编码] One-Hot 编码完成 (drop_first=True), 新列: {list(df.columns)}")
    return df


def main():
    # 解析命令行参数
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    # 检查输入文件是否存在
    if not input_path.exists():
        print(f"错误: 输入文件不存在 → {input_path}")
        sys.exit(1)

    # ---- 读取原始数据 ----
    print(f"读取数据: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  原始形状: {df.shape}")
    print(f"  列: {list(df.columns)}")

    # ---- Step 1: 处理缺失值（均值填充） ----
    print("\n[Step 1] 处理缺失值")
    df = handle_missing_values(df)

    # ---- Step 2: 缩尾处理（预算列的 99 分位数） ----
    print("\n[Step 2] 缩尾处理异常值")
    # 自动识别名称中包含 "Budget" 的列作为预算列
    budget_cols = [c for c in df.columns if "Budget" in c]
    df = winsorize_columns(df, budget_cols, quantile=0.99)

    # ---- Step 3: One-Hot 编码分类变量（drop_first=True） ----
    print("\n[Step 3] 编码分类变量")
    df = encode_categorical(df)

    # ---- 保存清洗后的数据 ----
    # 自动创建输出目录（如果不存在）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n清洗完成! 保存至: {output_path}")
    print(f"  最终形状: {df.shape}")
    print(f"  最终列: {list(df.columns)}")


if __name__ == "__main__":
    main()
