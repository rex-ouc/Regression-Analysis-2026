"""
入口程序:main.py
作用:导入模块，设置实验的超参数 (Hyperparameters)，并串联整个流水线。
执行方式:在终端运行 `uv run src/main.py`
"""

from components import loop, analysis


def main():
    """
    设计逻辑：配置隔离 (Configuration Isolation)。
    将所有可能变动的超参数全部集中在 main 函数开头。
    """
    print(">>> 实验开始：初始化超参数...")

    # 定义控制参数
    NUM_SIMULATIONS = 1000  # 模拟次数
    SAMPLE_SIZE = 100       # 每次抽样的样本量
    NOISE_STD = 1.0         # 噪音的标准差 (作业要求 epsilon ~ N(0,1))

    # 设定上帝视角的真实参数 [截距=1.0, 真实Beta1=2.0]
    TRUE_BETA = [1.0, 2.0]

    print(f">>> 开始执行 {NUM_SIMULATIONS} 次蒙特卡洛循环...")

    # 调用 simulation 模块中的 loop 函数
    results_df = loop(
        num_simulations=NUM_SIMULATIONS,
        sample_size=SAMPLE_SIZE,
        true_params=TRUE_BETA,
        noise_std=NOISE_STD,
    )

    print(">>> 循环结束，开始生成分析报告与图表...")

    # 调用 analysis 函数，传入真实 beta_1
    analysis(results_df, true_beta1=TRUE_BETA[1], output_file="src/week02/beta_dist.png")

    print(">>> 整个流水线执行完毕，请前往 Markdown 报告中查看结果！")


if __name__ == "__main__":
    main()