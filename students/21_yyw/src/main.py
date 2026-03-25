"""
入口程序：main.py
作用：导入模块，设置实验的超参数 (Hyperparameters)，并串联整个流水线。
执行方式：在终端运行 `uv run src/main.py` 或 `python main.py`
"""

# 【引入依赖】：从同级目录的 simulation.py 文件中，导入需要的功能模块
from simulation import loop, analysis


def main():
    """
    设计逻辑：配置隔离 (Configuration Isolation)
    
    将所有可能变动的超参数全部集中在 main 函数开头。
    如果明天要求做一套 10000 次模拟的新作业，只需要改这里的几个数字，
    而完全不需要去碰 simulation.py 里的任何底层代码。
    这叫"开闭原则"：对扩展开放，对修改封闭。
    """
    print("=" * 60)
    print(">>> 实验开始：初始化超参数...")
    print("=" * 60)

    # ========== 定义控制参数（常量大写表示品味） ==========
    NUM_SIMULATIONS = 1000      # 模拟次数
    SAMPLE_SIZE = 100           # 每次抽样的样本量
    NOISE_STD = 5.0             # 噪音的标准差

    # 设定上帝视角的真实参数 [截距, 斜率]
    TRUE_BETA = [3.0, 2.0]

    # 打印实验配置
    print(f"\n实验配置：")
    print(f"  - 模拟次数: {NUM_SIMULATIONS}")
    print(f"  - 样本量: {SAMPLE_SIZE}")
    print(f"  - 噪音标准差: {NOISE_STD}")
    print(f"  - 真实参数: β₀ = {TRUE_BETA[0]}, β₁ = {TRUE_BETA[1]}")

    print("\n>>> 开始执行蒙特卡洛循环...")

    # ========== 【调用模块】：核心执行阶段 ==========
    # 调用 simulation 模块中的 loop 函数
    results_df = loop(
        n_simulations=NUM_SIMULATIONS,
        n=SAMPLE_SIZE,
        true_params=TRUE_BETA,
        noise_std=NOISE_STD
    )

    print(f">>> 循环结束，共完成 {len(results_df)} 次模拟。")
    print(">>> 开始生成分析报告与图表...\n")

    # ========== 调用 simulation 模块中的 analysis 函数 ==========
    analysis(df_results=results_df, true_params=TRUE_BETA)

    print("\n" + "=" * 60)
    print(">>> 整个流水线执行完毕！")
    print(">>> 请查看以下输出：")
    print(">>>   - 终端中打印的 Markdown 表格")
    print(">>>   - assets/beta_dist.png 分布图")
    print("=" * 60)


# 【Python 经典工程惯例】
# 保护执行块：确保当此脚本被别人 import 时，不会意外运行 main()。
# 只有当在终端直接执行 `python main.py` 时，下面的代码才会生效。
if __name__ == "__main__":
    main()