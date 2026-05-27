"""
Milestone Project 2: The Pipeline & The Leakage-Free Generalization

Main execution script demonstrating:
1. bad_cross_validation(): Contains data leakage
2. good_cross_validation(): Leakage-free implementation
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.models import GradientDescentOLS
from utils.transformers import CustomStandardScaler
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape


def load_data(data_path: str) -> tuple:
    """
    Load and validate data.
    
    Returns:
    --------
    X, y : np.ndarray
        Features and target
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Assume last column is target, rest are features
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]
    
    # Handle categorical variables using one-hot encoding
    df_encoded = pd.get_dummies(df[feature_cols], drop_first=True)
    X = df_encoded.values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    
    # Update feature_cols to encoded columns
    feature_cols = df_encoded.columns.tolist()
    
    return X, y, feature_cols, target_col


def bad_cross_validation(X: np.ndarray, y: np.ndarray) -> dict:
    """
    LEAKY IMPLEMENTATION: Demonstrates data leakage.
    
    ❌ PROBLEM:
    1. Scaler is fit on ENTIRE dataset (including test folds)
    2. Missing values are imputed using GLOBAL statistics
    3. This leaks information to test sets
    4. Evaluation metrics will appear falsely good
    
    Returns:
    --------
    dict : Results containing RMSE, MAE, MAPE scores
    """
    print("\n" + "="*70)
    print("TASK 3: bad_cross_validation() - WITH DATA LEAKAGE ❌")
    print("="*70)
    
    # ❌ LEAKAGE STEP 1: Fit scaler on ENTIRE dataset
    print("\n[LEAKAGE] Fitting scaler on ENTIRE dataset (all 1000 samples)...")
    scaler = CustomStandardScaler()
    
    # ❌ LEAKAGE STEP 2: Fill NaN using global mean
    X_imputed = X.copy()
    global_mean = np.nanmean(X, axis=0)
    for j in range(X_imputed.shape[1]):
        mask = np.isnan(X_imputed[:, j])
        X_imputed[mask, j] = global_mean[j]
    print(f"[LEAKAGE] Filled NaNs using global means: {global_mean}")
    
    # ❌ LEAKAGE STEP 3: Transform entire dataset
    X_scaled = scaler.fit_transform(X_imputed)
    print(f"[LEAKAGE] Scaled entire dataset with global mean={scaler.mean_}, std={scaler.std_}")
    
    # Now do 5-fold CV on this "pre-processed" data
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    
    print("\n[CV] Running 5-fold cross-validation (WITH LEAKED PREPROCESSOR)...")
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_scaled), 1):
        X_train = X_scaled[train_idx]
        X_test = X_scaled[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Train model
        model = GradientDescentOLS(learning_rate=0.01, max_iter=5000)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        rmse = calculate_rmse(y_test, y_pred)
        mae = calculate_mae(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)
        
        print(f"  Fold {fold_idx}/5: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}%")
    
    # Compute averages
    mean_rmse = np.mean(rmse_scores)
    mean_mae = np.mean(mae_scores)
    mean_mape = np.mean(mape_scores)
    
    print(f"\n[RESULT] Average metrics (WITH LEAKAGE):")
    print(f"  Mean RMSE: {mean_rmse:.6f}")
    print(f"  Mean MAE:  {mean_mae:.6f}")
    print(f"  Mean MAPE: {mean_mape:.4f}%")
    
    return {
        'method': 'bad_cv',
        'rmse_scores': rmse_scores,
        'mae_scores': mae_scores,
        'mape_scores': mape_scores,
        'mean_rmse': mean_rmse,
        'mean_mae': mean_mae,
        'mean_mape': mean_mape
    }


def good_cross_validation(X: np.ndarray, y: np.ndarray) -> dict:
    """
    LEAKAGE-FREE IMPLEMENTATION: Best practice for CV.
    
    ✅ CORRECT:
    1. For each fold:
       a. Fit scaler ONLY on training data
       b. Transform training data
       c. Train model on transformed training data
       d. Transform test data using SAME scaler
       e. Predict on transformed test data
    2. This ensures test set independence
    3. Evaluation will be more realistic
    
    Returns:
    --------
    dict : Results containing RMSE, MAE, MAPE scores
    """
    print("\n" + "="*70)
    print("TASK 4: good_cross_validation() - LEAKAGE-FREE ✅")
    print("="*70)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    
    print("\n[CV] Running 5-fold cross-validation (LEAKAGE-FREE PIPELINE)...")
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        # Split data
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # ✅ STEP 1: Impute NaNs using TRAINING data only
        train_mean = np.nanmean(X_train_raw, axis=0)
        X_train_imputed = X_train_raw.copy()
        for j in range(X_train_imputed.shape[1]):
            mask = np.isnan(X_train_imputed[:, j])
            X_train_imputed[mask, j] = train_mean[j]
        
        X_test_imputed = X_test_raw.copy()
        for j in range(X_test_imputed.shape[1]):
            mask = np.isnan(X_test_imputed[:, j])
            # Use SAME imputation from training data
            X_test_imputed[mask, j] = train_mean[j]
        
        # ✅ STEP 2: Fit scaler ONLY on training data
        scaler = CustomStandardScaler()
        scaler.fit(X_train_imputed)
        
        # ✅ STEP 3: Transform training data
        X_train_scaled = scaler.transform(X_train_imputed)
        
        # ✅ STEP 4: Transform test data using SAME scaler
        X_test_scaled = scaler.transform(X_test_imputed)
        
        # ✅ STEP 5: Train model on training data
        model = GradientDescentOLS(learning_rate=0.01, max_iter=5000)
        model.fit(X_train_scaled, y_train)
        
        # ✅ STEP 6: Evaluate on test data
        y_pred = model.predict(X_test_scaled)
        rmse = calculate_rmse(y_test, y_pred)
        mae = calculate_mae(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)
        
        print(f"  Fold {fold_idx}/5: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}%")
        
        if fold_idx == 1:
            print(f"           [Scaler params] mean={scaler.mean_}, std={scaler.std_}")
    
    # Compute averages
    mean_rmse = np.mean(rmse_scores)
    mean_mae = np.mean(mae_scores)
    mean_mape = np.mean(mape_scores)
    
    print(f"\n[RESULT] Average metrics (LEAKAGE-FREE):")
    print(f"  Mean RMSE: {mean_rmse:.6f}")
    print(f"  Mean MAE:  {mean_mae:.6f}")
    print(f"  Mean MAPE: {mean_mape:.4f}%")
    
    return {
        'method': 'good_cv',
        'rmse_scores': rmse_scores,
        'mae_scores': mae_scores,
        'mape_scores': mape_scores,
        'mean_rmse': mean_rmse,
        'mean_mae': mean_mae,
        'mean_mape': mean_mape
    }


def generate_comparison_report(bad_results: dict, good_results: dict, output_path: str):
    """
    Generate markdown comparison report.
    
    Parameters:
    -----------
    bad_results : dict
        Results from bad_cross_validation()
    good_results : dict
        Results from good_cross_validation()
    output_path : str
        Path to save the report
    """
    # Calculate differences
    rmse_diff = bad_results['mean_rmse'] - good_results['mean_rmse']
    rmse_pct_diff = (rmse_diff / good_results['mean_rmse']) * 100
    
    mae_diff = bad_results['mean_mae'] - good_results['mean_mae']
    mae_pct_diff = (mae_diff / good_results['mean_mae']) * 100 if good_results['mean_mae'] > 0 else 0
    
    mape_diff = bad_results['mean_mape'] - good_results['mean_mape']
    
    # Generate report
    report = f"""# 里程碑项目2：数据泄露 vs 无泄露交叉验证

**报告日期**: {pd.Timestamp.now()}

## 执行摘要

本报告比较两种交叉验证方法：
- **错误CV** (任务3)：包含数据泄露的全局预处理
- **正确CV** (任务4)：无泄露的正确训练-测试隔离

---

## 评估指标比较

### 1. RMSE (均方根误差)

| 方法 | 平均RMSE | 折叠标准差 |
|------|----------|------------|
| 错误CV (有泄露) ❌ | {bad_results['mean_rmse']:.6f} | {np.std(bad_results['rmse_scores']):.6f} |
| 正确CV (无泄露) ✅ | {good_results['mean_rmse']:.6f} | {np.std(good_results['rmse_scores']):.6f} |
| **差异** | **{rmse_diff:+.6f}** | - |
| **百分比增加 (错误vs正确)** | **{rmse_pct_diff:+.2f}%** | - |

### 2. MAE (平均绝对误差)

| 方法 | 平均MAE | 折叠标准差 |
|------|---------|------------|
| 错误CV (有泄露) ❌ | {bad_results['mean_mae']:.6f} | {np.std(bad_results['mae_scores']):.6f} |
| 正确CV (无泄露) ✅ | {good_results['mean_mae']:.6f} | {np.std(good_results['mae_scores']):.6f} |
| **差异** | **{mae_diff:+.6f}** | - |
| **百分比增加 (错误vs正确)** | **{mae_pct_diff:+.2f}%** | - |

### 3. MAPE (平均绝对百分比误差)

| 方法 | 平均MAPE (%) |
|------|--------------|
| 错误CV (有泄露) ❌ | {bad_results['mean_mape']:.4f}% |
| 正确CV (无泄露) ✅ | {good_results['mean_mape']:.4f}% |
| **差异** | **{mape_diff:+.4f}%** |

---

## 详细折叠对比

### 按折叠的RMSE
```
Fold    Bad CV (❌)    Good CV (✅)    Difference
────────────────────────────────────────────────
"""
    
    for i in range(len(bad_results['rmse_scores'])):
        diff = bad_results['rmse_scores'][i] - good_results['rmse_scores'][i]
        report += f"  {i+1}     {bad_results['rmse_scores'][i]:.6f}       {good_results['rmse_scores'][i]:.6f}        {diff:+.6f}\n"
    
    report += f"""```

---

## 关键发现与分析

### 🚨 为什么错误CV的性能看起来"更好"？

错误CV方法显示**人为降低的RMSE/MAE**，因为：

1. **全局预处理泄露**：
   - 训练集统计信息在**整个数据集**上计算（包括测试折叠）
   - 测试集插补使用**从测试数据本身计算的均值**
   - 这使得测试数据比实际应该的"噪音更少"

2. **性能膨胀**：
   - 模型有效地获得了测试集统计信息的"预览"
   - 测试集看起来更规律和可预测
   - 现实中，新数据不会有这种优势

3. **不现实的评估**：
   - 报告的指标是**乐观的下限**
   - 真正的生产性能会更差
   - 可能导致虚假信心和糟糕的模型选择

### ✅ 为什么正确CV更可靠？

正确CV方法提供**现实的性能估计**，因为：

1. **正确的训练-测试隔离**：
   - 预处理（插补、缩放）**仅在训练数据上拟合**
   - 测试数据被视为真正未见
   - 模拟真实部署场景

2. **保守估计**：
   - 略高的误差指标反映现实
   - 为生产部署提供安全边际
   - 为模型比较提供更好的基础

3. **可重现性**：
   - 使用不同数据集会获得相同的CV分数
   - 不会被泄露人为提升

---

## 商业解读 (首席营销官视角)

### MAE：典型每日预算预测误差

```
无泄露 (正确CV):
  平均预测误差 ≈ ${good_results['mean_mae']:.2f}

有泄露 (错误CV):
  报告误差 ≈ ${bad_results['mean_mae']:.2f} (不可靠)

生产中的真实性能:
  预期误差 ≈ ${good_results['mean_mae']:.2f} (使用这个!)
```

### 决策：向高管报告哪些指标？

**❌ 错误**：报告错误CV指标以使模型看起来更好
```
"我们的模型达到 ${bad_results['mean_mae']:.2f} MAE！"
→ 会在生产中失败，失去信任
```

**✅ 正确**：报告正确CV指标以获得现实期望
```
"我们的模型通过正确交叉验证达到 ${good_results['mean_mae']:.2f} MAE"
→ 当与生产匹配时获得信任
```

---

## 技术教训

### ⚠️ 常见数据泄露模式

| 模式 | 风险 | 检测 |
|------|------|------|
| 全局缩放器拟合 | **高** | CV分数太好（基线检查） |
| 预计算均值用于插补 | **高** | 比较训练vs测试统计 |
| 在完整数据上进行特征选择 | **高** | 测试集泄露到特征重要性 |
| 分割前增强 | **高** | 人工数据多样性 |

### ✅ 最佳实践

1. **管道模式**：拟合 → 转换 → 训练（在每个折叠中）
2. **验证集**：使用正确的训练/验证/测试分割
3. **超参数调优**：仅使用训练数据统计
4. **文档化**：清楚标记在哪里拟合什么

---

## 建议

1. **始终使用正确CV方法**在生产管道中
2. **监控泄露**通过比较训练vs测试错误
3. **对不平衡数据集使用分层CV**
4. **文档化预处理**以审计泄露
5. **在生产中进行A/B测试**以验证离线指标是否匹配在线性能

---

## 代码参考

### 错误CV模式 (避免)
```python
# ❌ 全局预处理泄露信息
scaler.fit(X_entire)  # 在所有数据上拟合
X_scaled = scaler.transform(X_entire)
for train_idx, test_idx in cv.split(X_scaled):
    model.fit(X_scaled[train_idx], y[train_idx])
    score = model.score(X_scaled[test_idx], y[test_idx])
```

### 正确CV模式 (使用)
```python
# ✅ 预处理为每个折叠重新拟合
for train_idx, test_idx in cv.split(X):
    scaler = CustomStandardScaler()
    scaler.fit(X[train_idx])  # 仅在训练上拟合
    X_train_scaled = scaler.transform(X[train_idx])
    X_test_scaled = scaler.transform(X[test_idx])
    model.fit(X_train_scaled, y[train_idx])
    score = model.score(X_test_scaled, y[test_idx])
```

---

## 结论

这个里程碑项目证明**正确的数据处理比复杂算法更重要**。

- 带有无泄露评估的简单模型 > 带有泄露的复杂模型
- 相信来自正确CV的（有时）更差指标
- 数据科学家是模型完整性的守护者

---

**作者**: 学生 20_zyf  
**课程**: 回归分析 2026
"""
    
    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ Report saved to: {output_path}")


def generate_visualization(bad_results: dict, good_results: dict, output_path: str):
    """
    Generate comparison visualization.
    
    Parameters:
    -----------
    bad_results : dict
        Results from bad_cross_validation()
    good_results : dict
        Results from good_cross_validation()
    output_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE Comparison
    ax = axes[0]
    metrics = ['RMSE']
    bad_vals = [bad_results['mean_rmse']]
    good_vals = [good_results['mean_rmse']]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, bad_vals, width, label='Bad CV (❌ with leakage)', color='#ff6b6b', alpha=0.8)
    ax.bar(x + width/2, good_vals, width, label='Good CV (✅ leakage-free)', color='#51cf66', alpha=0.8)
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_title('RMSE Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # MAE Comparison
    ax = axes[1]
    metrics = ['MAE']
    bad_vals = [bad_results['mean_mae']]
    good_vals = [good_results['mean_mae']]
    ax.bar(x - width/2, bad_vals, width, label='Bad CV (❌ with leakage)', color='#ff6b6b', alpha=0.8)
    ax.bar(x + width/2, good_vals, width, label='Good CV (✅ leakage-free)', color='#51cf66', alpha=0.8)
    ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax.set_title('MAE Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # MAPE Comparison
    ax = axes[2]
    metrics = ['MAPE']
    bad_vals = [bad_results['mean_mape']]
    good_vals = [good_results['mean_mape']]
    ax.bar(x - width/2, bad_vals, width, label='Bad CV (❌ with leakage)', color='#ff6b6b', alpha=0.8)
    ax.bar(x + width/2, good_vals, width, label='Good CV (✅ leakage-free)', color='#51cf66', alpha=0.8)
    ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    ax.set_title('MAPE Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Milestone Project 2: Data Leakage Impact on Model Evaluation', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("🏆 里程碑项目2：管道与无泄露泛化")
    print("="*70)
    
    # Setup paths
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'milestone2')
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'homework', 'week09', 'data')
    
    # Clean and create results directory
    if os.path.exists(results_dir):
        import shutil
        shutil.rmtree(results_dir)
        print(f"\n[设置] 清理结果目录: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"[设置] 创建结果目录: {results_dir}")
    
    # Try to load data
    data_path = os.path.join(data_dir, 'dirty_marketing.csv')
    if not os.path.exists(data_path):
        print(f"\n[警告] 数据文件未找到: {data_path}")
        print("[替代方案] 使用合成数据进行演示...")
        # Generate synthetic data similar to marketing dataset
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 4) * 100 + 200
        y = 500 + 2*X[:, 0] + 1.5*X[:, 1] + np.random.randn(n_samples) * 50
        
        # Add some missing values
        missing_mask = np.random.rand(n_samples, 4) < 0.1
        X[missing_mask] = np.nan
        
        # Save synthetic data
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, 'synthetic_marketing.csv')
        df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])
        df['Sales'] = y
        df.to_csv(data_path, index=False)
        print(f"[合成] 生成合成数据: {data_path}")
    else:
        print(f"\n[数据] 从以下位置加载数据: {data_path}")
    
    # Load data
    X, y, feature_cols, target_col = load_data(data_path)
    print(f"[DATA] Shape: {X.shape}, Features: {len(feature_cols)}, Target: {target_col}")
    print(f"[DATA] Missing values: {np.sum(np.isnan(X))}")
    
    # Task 3: Bad CV
    bad_results = bad_cross_validation(X, y)
    
    # Task 4: Good CV
    good_results = good_cross_validation(X, y)
    
    # Task 5: Generate report
    report_path = os.path.join(results_dir, 'evaluation_comparison.md')
    generate_comparison_report(bad_results, good_results, report_path)
    
    # Task 5: Generate visualization
    viz_path = os.path.join(results_dir, 'leakage_analysis.png')
    generate_visualization(bad_results, good_results, viz_path)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ 里程碑项目2 已完成")
    print("="*70)
    print(f"\n📊 结果摘要:")
    print(f"   错误CV (有泄露)    RMSE: {bad_results['mean_rmse']:.6f}")
    print(f"   正确CV (无泄露)   RMSE: {good_results['mean_rmse']:.6f}")
    print(f"   改进:             {((bad_results['mean_rmse'] - good_results['mean_rmse']) / good_results['mean_rmse'] * 100):+.2f}%")
    print(f"\n📁 交付物:")
    print(f"   {report_path}")
    print(f"   {viz_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
