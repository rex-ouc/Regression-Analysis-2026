"""
Main program for linear regression solver comparison.
"""

import numpy as np  #数值计算库
import pandas as pd  #数据分析库
import sys  #系统相关功能，用于路径操作
import os  #操作系统接口，用于文件和目录操作
import time  #时间相关，用于性能计时

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_data  #生成合成数据
from compare_methods import run_all_comparisons  #运行所有求解器对比


def main():
    print("\n" + "="*80)
    print("The Tale of Two Solvers - Linear Regression Comparison")
    print("="*80)
    
    N = 10000  #样本数量
    noise_std = 0.1  #噪声标准差，控制数据中随机噪声的大小
    
    # Experiment A: Low dimension
    print("\n" + "="*80)
    print("Experiment A: Low-dimensional (P=10)")  #10个特征
    print("="*80)
    
    print("Generating data...")
    X_low, y_low, _ = generate_data(N, 10, noise_std)  #X_low：特征矩阵，y_low：目标向量
    
    print("Running solvers...")
    results_low = run_all_comparisons(X_low, y_low, "Low-dim (N=10000, P=10)")  #调用 run_all_comparisons 运行所有5个求解器
    
    # Experiment B: High dimension
    print("\n" + "="*80)
    print("Experiment B: High-dimensional (P=200)")
    print("="*80)
    
    print("Generating data...")
    X_high, y_high, _ = generate_data(N, 200, noise_std)
    
    print("Running solvers...")
    results_high = run_all_comparisons(X_high, y_high, "High-dim (N=10000, P=200)")
    
    # Combine results
    all_results = pd.concat([results_low, results_high], ignore_index=True)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    display_df = all_results[['scenario', 'method', 'time', 'mse']].copy()
    display_df['time'] = display_df['time'].apply(lambda x: f"{x:.4f}s" if pd.notna(x) else "N/A")
    display_df['mse'] = display_df['mse'].apply(lambda x: f"{x:.6e}" if pd.notna(x) else "N/A")
    print(display_df.to_string(index=False))
    
    # Time comparison
    print("\n" + "="*80)
    print("TIME COMPARISON (seconds)")
    print("="*80)
    pivot_time = all_results.pivot_table(index='scenario', columns='method', values='time', aggfunc='first')
    print(pivot_time.round(6))
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'comparison_results.csv')
    all_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Find fastest
    high_res = results_high[results_high['time'].notna()]
    if len(high_res) > 0:
        fastest = high_res.loc[high_res['time'].idxmin()]
        print(f"\n✓ Fastest solver in high dimension: {fastest['method']} ({fastest['time']:.4f}s)")
    
    print("\n" + "="*80)
    print("Why SGDRegressor is fastest in high dimensions?")
    print("="*80)
    print("1. Complexity: O(P) vs O(P³)")
    print("2. Memory: Stores P elements vs P×P matrix")
    print("3. Uses stochastic updates and adaptive learning rates")
    
    return all_results


if __name__ == "__main__":
    results = main()