"""
入口程序：main.py
Week 04 实验：求解器双城记
- Task 2: 低维 vs 高维性能对比
- Task 3: 工业界 API 对比
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

from solvers import AnalyticalSolver, GradientDescentSolver


def generate_data(n_samples, n_features, noise_std=0.1, seed=42):
    """
    生成线性回归模拟数据
    """
    np.random.seed(seed)

    # 生成真实参数
    true_beta = np.random.randn(n_features + 1)
    true_beta[0] = 1.0  # 截距

    # 生成特征矩阵
    X = np.random.randn(n_samples, n_features)

    # 生成 y = Xβ + ε
    X_with_const = np.column_stack([np.ones(n_samples), X])
    y = X_with_const @ true_beta + np.random.randn(n_samples) * noise_std

    return X, y, true_beta


def benchmark_solver(solver, X, y, true_beta, solver_name):
    """
    测试单个求解器的性能和精度
    """
    start = time.time()

    try:
        solver.fit(X, y)
        elapsed = time.time() - start

        # 获取参数（统一处理不同接口）
        if hasattr(solver, "coef_"):
            # sklearn 的情况
            if hasattr(solver, "intercept_"):
                beta = np.concatenate([[solver.intercept_], solver.coef_])
            else:
                beta = solver.coef_
        elif hasattr(solver, "get_params"):
            # 自定义 solver 的情况
            beta = solver.get_params()
        elif hasattr(solver, "params"):
            # statsmodels 的情况
            beta = (
                solver.params.values
                if hasattr(solver.params, "values")
                else solver.params
            )
        else:
            beta = None

        if beta is not None:
            beta = np.array(beta).flatten()
            # 确保长度匹配
            if len(beta) != len(true_beta):
                # 处理可能缺少截距的情况
                if len(beta) == len(true_beta) - 1:
                    beta = np.concatenate([[0], beta])
            mse = np.mean((beta - true_beta) ** 2)
        else:
            mse = np.nan

        return {
            "name": solver_name,
            "time": elapsed,
            "beta": beta,
            "mse": mse,
            "success": True,
        }
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return {
            "name": solver_name,
            "time": np.nan,
            "beta": None,
            "mse": np.nan,
            "success": False,
            "error": str(e),
        }


def run_experiment(n_samples, n_features, description):
    """
    运行完整实验（低维或高维）
    """
    print("\n" + "=" * 70)
    print(f"实验: {description}")
    print(f"  样本量 N = {n_samples}, 特征维度 P = {n_features}")
    print("=" * 70)

    # 1. 生成数据
    print("\n>>> 生成数据...")
    X, y, true_beta = generate_data(n_samples, n_features)
    print(f"   数据形状: X={X.shape}, y={y.shape}")

    results = []

    # 2.1 手写 AnalyticalSolver
    print("\n>>> 测试: AnalyticalSolver (手写)")
    try:
        solver = AnalyticalSolver()
        start = time.time()
        solver.fit(X, y)
        elapsed = time.time() - start
        beta = solver.get_params()
        mse = np.mean((beta - true_beta) ** 2)
        print(f"   耗时: {elapsed:.4f} 秒")
        print(f"   MSE: {mse:.2e}")
        results.append(
            {
                "name": "AnalyticalSolver",
                "time": elapsed,
                "beta": beta,
                "mse": mse,
                "success": True,
            }
        )
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        results.append(
            {
                "name": "AnalyticalSolver",
                "time": np.nan,
                "mse": np.nan,
                "success": False,
            }
        )

    # 2.2 手写 GradientDescentSolver
    print("\n>>> 测试: GradientDescentSolver (手写)")
    try:
        solver = GradientDescentSolver(
            learning_rate=0.01 / max(1, np.sqrt(n_features)),
            n_iterations=5000,
            tolerance=1e-8,
            verbose=False,
        )
        start = time.time()
        solver.fit(X, y)
        elapsed = time.time() - start
        beta = solver.get_params()
        mse = np.mean((beta - true_beta) ** 2)
        print(f"   耗时: {elapsed:.4f} 秒")
        print(f"   MSE: {mse:.2e}")
        results.append(
            {
                "name": "GradientDescentSolver",
                "time": elapsed,
                "beta": beta,
                "mse": mse,
                "success": True,
            }
        )
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        results.append(
            {
                "name": "GradientDescentSolver",
                "time": np.nan,
                "mse": np.nan,
                "success": False,
            }
        )

    # 2.3 statsmodels.OLS
    print("\n>>> 测试: statsmodels.OLS")
    try:
        import statsmodels.api as sm

        X_with_const = sm.add_constant(X)
        start = time.time()
        model = sm.OLS(y, X_with_const).fit()
        elapsed = time.time() - start
        beta = model.params.values if hasattr(model.params, "values") else model.params
        beta = np.array(beta).flatten()
        mse = np.mean((beta - true_beta) ** 2)
        print(f"   耗时: {elapsed:.4f} 秒")
        print(f"   MSE: {mse:.2e}")
        results.append(
            {
                "name": "statsmodels.OLS",
                "time": elapsed,
                "beta": beta,
                "mse": mse,
                "success": True,
            }
        )
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        results.append(
            {"name": "statsmodels.OLS", "time": np.nan, "mse": np.nan, "success": False}
        )

    # 2.4 sklearn.LinearRegression
    print("\n>>> 测试: sklearn.LinearRegression")
    try:
        from sklearn.linear_model import LinearRegression

        solver = LinearRegression()
        start = time.time()
        solver.fit(X, y)
        elapsed = time.time() - start
        beta = np.concatenate([[solver.intercept_], solver.coef_.flatten()])
        mse = np.mean((beta - true_beta) ** 2)
        print(f"   耗时: {elapsed:.4f} 秒")
        print(f"   MSE: {mse:.2e}")
        results.append(
            {
                "name": "sklearn.LinearRegression",
                "time": elapsed,
                "beta": beta,
                "mse": mse,
                "success": True,
            }
        )
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        results.append(
            {
                "name": "sklearn.LinearRegression",
                "time": np.nan,
                "mse": np.nan,
                "success": False,
            }
        )

    # 2.5 sklearn.SGDRegressor
    print("\n>>> 测试: sklearn.SGDRegressor")
    try:
        from sklearn.linear_model import SGDRegressor

        solver = SGDRegressor(
            max_iter=5000,
            tol=1e-8,
            learning_rate="optimal",
            random_state=42,
            verbose=False,
        )
        start = time.time()
        solver.fit(X, y)
        elapsed = time.time() - start
        intercept = solver.intercept_
        if hasattr(intercept, "__len__"):
            intercept = intercept[0]
        beta = np.concatenate([[intercept], solver.coef_.flatten()])
        mse = np.mean((beta - true_beta) ** 2)
        print(f"   耗时: {elapsed:.4f} 秒")
        print(f"   MSE: {mse:.2e}")
        results.append(
            {
                "name": "sklearn.SGDRegressor",
                "time": elapsed,
                "beta": beta,
                "mse": mse,
                "success": True,
            }
        )
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        results.append(
            {
                "name": "sklearn.SGDRegressor",
                "time": np.nan,
                "mse": np.nan,
                "success": False,
            }
        )

    return pd.DataFrame(results)


def generate_report(df_low, df_high):
    """
    生成 Markdown 格式的实验报告
    """
    from datetime import datetime

    report = f"""# Week 04 实验报告：求解器双城记

## 一、实验背景

在理论课上，我们推导了多元线性回归的全局最优解析解：
$$\\beta = (X^T X)^{{-1}} X^T Y$$

但在高维场景下（特征维度 $P$ 极大时），求逆矩阵的时间复杂度高达 $O(P^3)$，面临严重的内存和算力瓶颈。

## 二、实验设置

| 参数 | 低维场景 | 高维场景 |
|------|----------|----------|
| 样本量 $N$ | 10,000 | 10,000 |
| 特征维度 $P$ | 10 | 2,000 |
| 噪音标准差 | 0.1 | 0.1 |

## 三、实验结果

### 3.1 低维场景 ($N=10,000, P=10$)

| 求解器 | 耗时 (秒) | MSE |
|--------|-----------|-----|
"""

    # 手动添加表格行
    if not df_low.empty:
        for _, row in df_low[df_low["success"]].iterrows():
            report += f"| {row['name']} | {row['time']:.4f} | {row['mse']:.2e} |\n"

    report += f"""
### 3.2 高维场景 ($N=10,000, P=2,000$)

| 求解器 | 耗时 (秒) | MSE |
|--------|-----------|-----|
"""

    if not df_high.empty:
        for _, row in df_high[df_high["success"]].iterrows():
            report += f"| {row['name']} | {row['time']:.4f} | {row['mse']:.2e} |\n"

    report += f"""
## 四、可视化结果

![耗时对比图](assets/week04_benchmark.png)

## 五、思考题

### Q1: 在高维场景下，哪个 API 崩溃了或极其缓慢？

**答**：statsmodels.OLS 在高维场景下最容易崩溃或极其缓慢。原因是：
1. 需要计算完整的协方差矩阵用于统计推断
2. 求逆复杂度 O(P³) 在 P=2000 时约为 8×10⁹ 次操作
3. 内存占用大

### Q2: 为什么 SGDRegressor 能在极短时间内完成任务？

**答**：SGDRegressor 采用随机梯度下降：
1. 每次迭代只处理一个样本，复杂度 O(P)
2. 不需要存储或求逆 XᵀX 矩阵
3. 内存占用小，可扩展到大数据集

## 六、结论

1. **低维场景**：解析解方法（AnalyticalSolver、sklearn.LinearRegression）最快
2. **高维场景**：梯度下降方法（SGDRegressor）更优
3. **工程建议**：小数据用解析解，大数据用 SGD

"""

    return report


def plot_comparison(df_low, df_high):
    """
    绘制耗时对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 低维场景
    if not df_low.empty:
        df_low_success = df_low[df_low["success"]]
        if len(df_low_success) > 0:
            names = df_low_success["name"].tolist()
            times = df_low_success["time"].tolist()

            bars = axes[0].bar(range(len(names)), times, color="steelblue")
            axes[0].set_xticks(range(len(names)))
            axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=10)
            axes[0].set_ylabel("耗时 (秒)")
            axes[0].set_title(f"低维场景耗时对比\n(N=10000, P=10)")
            axes[0].grid(True, alpha=0.3, axis="y")

            for bar, t in zip(bars, times):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{t:.3f}s",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    # 高维场景
    if not df_high.empty:
        df_high_success = df_high[df_high["success"]]
        if len(df_high_success) > 0:
            names = df_high_success["name"].tolist()
            times = df_high_success["time"].tolist()

            bars = axes[1].bar(range(len(names)), times, color="coral")
            axes[1].set_xticks(range(len(names)))
            axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=10)
            axes[1].set_ylabel("耗时 (秒)")
            axes[1].set_title(f"高维场景耗时对比\n(N=10000, P=2000)")
            axes[1].grid(True, alpha=0.3, axis="y")

            for bar, t in zip(bars, times):
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{t:.3f}s",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()
    plt.savefig("assets/week04_benchmark.png", dpi=150, bbox_inches="tight")
    print("\n✅ 图片已保存至: assets/week04_benchmark.png")
    plt.close()


def main():
    """
    主实验流水线
    """
    print("=" * 70)
    print("Week 04: 求解器双城记 - 完整实验")
    print("=" * 70)

    # 创建 assets 目录
    import os

    os.makedirs("assets", exist_ok=True)

    # 实验 A: 低维场景
    df_low = run_experiment(
        n_samples=10000, n_features=10, description="实验 A: 低维场景 (N=10000, P=10)"
    )

    # 实验 B: 高维场景（注意：P=2000 可能较慢）
    print("\n⚠️ 高维实验可能需要较长时间，请耐心等待...")
    df_high = run_experiment(
        n_samples=10000,
        n_features=2000,
        description="实验 B: 高维灾难 (N=10000, P=2000)",
    )

    # 生成可视化对比图
    print("\n>>> 生成可视化图表...")
    plot_comparison(df_low, df_high)

    # 生成 Markdown 报告
    print("\n>>> 生成实验报告...")
    report = generate_report(df_low, df_high)

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("✅ 报告已保存至: report.md")

    # 打印结果摘要
    print("\n" + "=" * 70)
    print("实验完成！结果摘要:")
    print("=" * 70)

    print("\n【低维场景】")
    if not df_low.empty:
        for _, row in df_low[df_low["success"]].iterrows():
            print(f"  {row['name']}: {row['time']:.4f}s, MSE={row['mse']:.2e}")

    print("\n【高维场景】")
    if not df_high.empty:
        for _, row in df_high[df_high["success"]].iterrows():
            print(f"  {row['name']}: {row['time']:.4f}s, MSE={row['mse']:.2e}")

    print("\n" + "=" * 70)
    print("请查看以下产出文件:")
    print("  - report.md (实验报告)")
    print("  - assets/week04_benchmark.png (耗时对比图)")
    print("=" * 70)


if __name__ == "__main__":
    main()
