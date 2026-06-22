#!/usr/bin/env python3
"""
Week 15: Logistic Regression and Binary Classification
完整版：Task A-E（模拟数据 + 真实数据）
Usage: uv run src/week15/main.py
"""

import sys
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, log_loss,
                             roc_curve)

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.metrics import calculate_rmse, calculate_mae

# 设置绘图
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# ========================== Task A: 生成二分类数据 ==========================
def generate_binary_data(n_samples=500, n_features=4, random_seed=42):
    """生成带有明确概率结构的二分类数据"""
    np.random.seed(random_seed)
    
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # 真实系数：X1增加概率，X2减少概率，X3和X4为噪声
    beta = np.array([2.0, -1.5, 0.0, 0.0])
    intercept = 0.5
    
    eta = intercept + X @ beta
    p = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, p)
    
    feature_names = [f'X{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['p_true'] = p
    df['y'] = y
    
    metadata = {
        'n_samples': n_samples,
        'n_features': n_features,
        'beta': beta.tolist(),
        'intercept': intercept,
        'description': 'X1 increases P(y=1), X2 decreases P(y=1), X3 and X4 are noise'
    }
    return df, metadata


def generate_highdim_binary_data(n_samples=400, n_features=25, random_seed=42):
    """生成高维二分类数据（含相关特征）用于 Task D"""
    np.random.seed(random_seed)
    
    latent = np.random.normal(0, 1, n_samples)
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # 前5个特征高度相关（共线性）
    for i in range(5):
        X[:, i] = 0.8 * latent + 0.2 * np.random.normal(0, 1, n_samples)
    
    # 只有前3个特征真正影响 y
    beta = np.zeros(n_features)
    beta[:3] = [2.0, -1.5, 1.0]
    intercept = 0.3
    
    eta = intercept + X @ beta
    p = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, p)
    
    feature_names = [f'X{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['y'] = y
    return df, feature_names


# ========================== Task A4: 对比图 ==========================
def plot_linear_vs_logistic(df, feature_names, output_path):
    """绘制 LinearRegression vs LogisticRegression 对比图"""
    X = df[feature_names].values
    y = df['y'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    
    logreg = LogisticRegression()
    logreg.fit(X_train_scaled, y_train)
    
    # 使用第一个特征做可视化
    X1_test = X_test[:, 0].reshape(-1, 1)
    X1_test_scaled = scaler.transform(np.column_stack([X1_test, np.zeros((len(X1_test), X.shape[1]-1))]))[:, 0]
    
    # 生成平滑曲线
    x_sorted = np.sort(X1_test_scaled)
    x_plot = np.linspace(x_sorted.min(), x_sorted.max(), 100)
    
    # 构造完整特征矩阵用于预测
    X_plot = np.zeros((100, X.shape[1]))
    X_plot[:, 0] = x_plot
    X_plot_scaled = scaler.transform(X_plot)
    
    y_lr_plot = lr.predict(X_plot_scaled)
    y_logreg_plot = logreg.predict_proba(X_plot_scaled)[:, 1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 散点图
    colors = ['blue' if yi == 0 else 'red' for yi in y_test]
    ax.scatter(X1_test, y_test, c=colors, alpha=0.5, label='True Labels')
    
    # 线性回归预测
    ax.plot(x_plot, y_lr_plot, 'g--', linewidth=2, label='Linear Regression')
    
    # 逻辑回归预测
    ax.plot(x_plot, y_logreg_plot, 'b-', linewidth=2, label='Logistic Regression')
    
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Threshold=0.5')
    ax.set_xlabel('Feature X1 (standardized)')
    ax.set_ylabel('Predicted Value / Probability')
    ax.set_title('Linear Regression vs Logistic Regression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  📊 Comparison plot saved: {output_path}")


# ========================== Task B: Loss 曲线 ==========================
def plot_loss_curves(output_path):
    """绘制 log loss vs squared error 对比图"""
    p = np.linspace(0.001, 0.999, 100)
    
    # y=1 时的损失
    log_loss_y1 = -np.log(p)
    mse_y1 = (1 - p) ** 2
    
    # y=0 时的损失
    log_loss_y0 = -np.log(1 - p)
    mse_y0 = p ** 2
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # y=1
    axes[0].plot(p, log_loss_y1, 'b-', linewidth=2, label='Log Loss')
    axes[0].plot(p, mse_y1, 'r--', linewidth=2, label='Squared Error')
    axes[0].set_xlabel('Predicted Probability p')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss when y=1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # y=0
    axes[1].plot(p, log_loss_y0, 'b-', linewidth=2, label='Log Loss')
    axes[1].plot(p, mse_y0, 'r--', linewidth=2, label='Squared Error')
    axes[1].set_xlabel('Predicted Probability p')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss when y=0')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  📊 Loss curves saved: {output_path}")


# ========================== Task C: Threshold 扫描 ==========================
def threshold_scan(y_true, y_prob, thresholds):
    """扫描不同阈值下的分类指标"""
    results = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        results.append({
            'threshold': thresh,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        })
    return pd.DataFrame(results)


def plot_threshold_curves(df_threshold, output_path):
    """绘制 threshold 曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df_threshold['threshold'], df_threshold['accuracy'], 'b-o', label='Accuracy')
    ax.plot(df_threshold['threshold'], df_threshold['precision'], 'g-s', label='Precision')
    ax.plot(df_threshold['threshold'], df_threshold['recall'], 'r-^', label='Recall')
    ax.plot(df_threshold['threshold'], df_threshold['f1'], 'm-d', label='F1')
    
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title('Threshold vs Classification Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  📊 Threshold curves saved: {output_path}")


# ========================== Task D: 正则化比较 ==========================
def compare_l1_l2_logistic(X_train, y_train, X_test, y_test):
    """比较 L1 和 L2 正则化逻辑回归"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    param_grid = {'C': np.logspace(-3, 2, 20)}
    
    # L1 正则化 - 使用 saga 求解器
    print("\n  [L1 Regularization]")
    # 注意：penalty='elasticnet' + l1_ratio=1 等价于 L1
    grid_l1 = GridSearchCV(
        LogisticRegression(
            penalty='elasticnet', 
            l1_ratio=1,           # 1 表示纯 L1
            solver='saga',         # saga 是唯一支持 elasticnet 的求解器
            max_iter=5000
        ),
        param_grid, cv=5, scoring='accuracy'
    )
    grid_l1.fit(X_train_scaled, y_train)
    best_l1 = grid_l1.best_estimator_
    print(f"    Best C: {grid_l1.best_params_['C']:.4f}")
    
    y_pred_l1 = best_l1.predict(X_test_scaled)
    y_prob_l1 = best_l1.predict_proba(X_test_scaled)[:, 1]
    
    results['L1'] = {
        'accuracy': accuracy_score(y_test, y_pred_l1),
        'precision': precision_score(y_test, y_pred_l1),
        'recall': recall_score(y_test, y_pred_l1),
        'f1': f1_score(y_test, y_pred_l1),
        'roc_auc': roc_auc_score(y_test, y_prob_l1),
        'log_loss': log_loss(y_test, y_prob_l1),
        'n_nonzero': np.sum(np.abs(best_l1.coef_) > 1e-6)
    }
    
    # L2 正则化
    print("\n  [L2 Regularization]")
    # 方式1: 使用新的 elasticnet 写法
    grid_l2 = GridSearchCV(
        LogisticRegression(
            penalty='elasticnet',
            l1_ratio=0,           # 0 表示纯 L2
            solver='saga',
            max_iter=5000
        ),
        param_grid, cv=5, scoring='accuracy'
    )
    # 或者方式2: 使用传统的 penalty='l2'（更简洁，推荐）
    # grid_l2 = GridSearchCV(
    #     LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000),
    #     param_grid, cv=5, scoring='accuracy'
    # )
    grid_l2.fit(X_train_scaled, y_train)
    best_l2 = grid_l2.best_estimator_
    print(f"    Best C: {grid_l2.best_params_['C']:.4f}")
    
    y_pred_l2 = best_l2.predict(X_test_scaled)
    y_prob_l2 = best_l2.predict_proba(X_test_scaled)[:, 1]
    
    results['L2'] = {
        'accuracy': accuracy_score(y_test, y_pred_l2),
        'precision': precision_score(y_test, y_pred_l2),
        'recall': recall_score(y_test, y_pred_l2),
        'f1': f1_score(y_test, y_pred_l2),
        'roc_auc': roc_auc_score(y_test, y_prob_l2),
        'log_loss': log_loss(y_test, y_prob_l2),
        'n_nonzero': X_train.shape[1]  # L2 保留所有特征
    }
    
    return results, best_l1, best_l2

def plot_regularization_comparison(results, output_path):
    """绘制正则化对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 性能指标
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    x = np.arange(len(metrics))
    width = 0.35
    
    l1_vals = [results['L1'][m] for m in metrics]
    l2_vals = [results['L2'][m] for m in metrics]
    
    axes[0].bar(x - width/2, l1_vals, width, label='L1', color='skyblue')
    axes[0].bar(x + width/2, l2_vals, width, label='L2', color='salmon')
    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Performance: L1 vs L2')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 非零系数个数
    n_nonzero = [results['L1']['n_nonzero'], results['L2']['n_nonzero']]
    axes[1].bar(['L1', 'L2'], n_nonzero, color=['skyblue', 'salmon'])
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Number of Non-zero Coefficients')
    axes[1].set_title('Model Complexity: L1 vs L2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  📊 Regularization comparison saved: {output_path}")


# ========================== Task E: 真实数据 ==========================
def load_and_preprocess_real_data(data_dir):
    """加载并预处理电信客户流失数据（修复 NaN 问题）"""
    data_path = Path(data_dir) / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"请将 Telco 数据放在 {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✅ 加载真实数据: {data_path}, 形状: {df.shape}")
    
    # 1. 删除无意义的 ID 列
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    # 2. 安全地处理 TotalCharges（转换为数值，用中位数填补）
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # 使用非 inplace 的赋值方式
    median_total_charges = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)
    
    # 3. 处理目标变量 Churn
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # 4. 处理所有非数值列（类别变量编码）
    cat_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # 5. 分离特征和目标
    X = df.drop('Churn', axis=1).values
    y = df['Churn'].values
    
    # 6. 彻底检查并处理可能残留的 NaN（防御性编程）
    if np.any(np.isnan(X)):
        print(f"  ⚠️ 警告: 特征矩阵中仍有 {np.isnan(X).sum()} 个 NaN，将用 0 填充。")
        X = np.nan_to_num(X, nan=0.0)
    
    print(f"  特征数: {X.shape[1]}, 正类比例: {y.mean():.4f}")
    return X, y, df.columns.tolist()

def run_real_data_task(data_dir, results_dir):
    """运行真实数据任务"""
    print("\n" + "="*70)
    print("Task E: Real Data - Telco Customer Churn")
    print("="*70)
    
    X, y, feature_names = load_and_preprocess_real_data(data_dir)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练逻辑回归
    print("\n[1] Training Logistic Regression")
    logreg = LogisticRegression(max_iter=5000)
    logreg.fit(X_train_scaled, y_train)
    
    y_prob = logreg.predict_proba(X_test_scaled)[:, 1]
    y_pred = logreg.predict(X_test_scaled)
    
    # 基础指标
    print("\n[2] Confusion Matrix & Metrics")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n{cm}")
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Threshold 扫描
    print("\n[3] Threshold Analysis")
    thresholds = np.arange(0.1, 1.0, 0.1)
    df_threshold = threshold_scan(y_test, y_prob, thresholds)
    print(df_threshold.to_string(index=False))
    plot_threshold_curves(df_threshold, Path(results_dir) / "real_threshold_curves.png")
    
    # L1 vs L2 比较（可选）
    print("\n[4] L1 vs L2 Regularization")
    reg_results, _, _ = compare_l1_l2_logistic(X_train, y_train, X_test, y_test)
    plot_regularization_comparison(reg_results, Path(results_dir) / "real_regularization_comparison.png")
    
    # 生成报告
    report_path = Path(results_dir) / "real_data_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Real Data Report: Telco Customer Churn\n\n")
        f.write("## Dataset Info\n")
        f.write("- Source: Kaggle Telco Customer Churn\n")
        f.write(f"- Samples: {len(y)}, Features: {X.shape[1]}\n")
        f.write(f"- Positive class ratio (Churn): {y.mean():.4f}\n\n")
        
        f.write("## Confusion Matrix\n")
        f.write(f"```\n{cm}\n```\n\n")
        
        f.write("## Performance Metrics\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for k, v in metrics.items():
            f.write(f"| {k} | {v:.4f} |\n")
        f.write("\n")
        
        f.write("## Threshold Analysis\n")
        f.write("![Threshold Curves](real_threshold_curves.png)\n\n")
        
        f.write("## L1 vs L2 Regularization\n")
        f.write("![Regularization Comparison](real_regularization_comparison.png)\n\n")
        
        f.write("## Business Interpretation\n")
        f.write("1. **Accuracy alone is misleading**: With ~73% non-churn, a dumb model could get 73% accuracy.\n")
        f.write("2. **Most trusted metric**: Recall, because missing a churner (FN) is more costly than false alarm (FP).\n")
        f.write("3. **Probability over class**: Emphasize probability to business teams for risk-based decision making.\n")
    
    print(f"📄 Report: {report_path}")
    return report_path


# ========================== 模拟数据主流程 ==========================
def run_synthetic_task(data_dir, results_dir):
    print("\n" + "="*70)
    print("Task A-D: Synthetic Data - Logistic Regression Experiments")
    print("="*70)
    
    # A1: 生成数据
    data_path = Path(data_dir) / "synthetic_binary.csv"
    df, metadata = generate_binary_data(n_samples=500, n_features=4)
    df.to_csv(data_path, index=False)
    print(f"✅ Synthetic data generated: {data_path}")
    print(f"   Samples: {metadata['n_samples']}, Features: {metadata['n_features']}")
    print(f"   DGP: {metadata['description']}")
    
    feature_names = [col for col in df.columns if col not in ['p_true', 'y']]
    X, y = df[feature_names].values, df['y'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # A3: 模型对比
    print("\n[Task A] Linear Regression vs Logistic Regression")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    
    logreg = LogisticRegression()
    logreg.fit(X_train_scaled, y_train)
    y_prob_logreg = logreg.predict_proba(X_test_scaled)[:, 1]
    y_pred_logreg = logreg.predict(X_test_scaled)
    
    print(f"  Linear Regression - Test RMSE: {calculate_rmse(y_test, y_pred_lr):.4f}")
    print(f"  Logistic Regression - Test Accuracy: {accuracy_score(y_test, y_pred_logreg):.4f}")
    
    # A4: 对比图
    plot_linear_vs_logistic(df, feature_names, Path(results_dir) / "linear_vs_logistic.png")
    
    # B2: Loss 曲线
    print("\n[Task B] Loss Curves")
    plot_loss_curves(Path(results_dir) / "loss_curves.png")
    
    # C: Threshold 扫描
    print("\n[Task C] Threshold Analysis")
    thresholds = np.arange(0.1, 1.0, 0.1)
    df_threshold = threshold_scan(y_test, y_prob_logreg, thresholds)
    print(df_threshold.to_string(index=False))
    plot_threshold_curves(df_threshold, Path(results_dir) / "threshold_curves.png")
    
    # D: 正则化比较
    print("\n[Task D] L1 vs L2 Regularization")
    reg_results, _, _ = compare_l1_l2_logistic(X_train, y_train, X_test, y_test)
    plot_regularization_comparison(reg_results, Path(results_dir) / "regularization_comparison.png")
    
    # 生成报告
    generate_synthetic_report(results_dir, df, metadata, y_test, y_pred_logreg, y_prob_logreg)
    generate_threshold_report(results_dir, df_threshold)
    generate_regularization_report(results_dir, reg_results)
    
    return df


def generate_synthetic_report(results_dir, df, metadata, y_test, y_pred, y_prob):
    report_path = Path(results_dir) / "synthetic_report.md"
    cm = confusion_matrix(y_test, y_pred)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Synthetic Data: Logistic Regression Report\n\n")
        f.write("## 1. Data Generation (DGP)\n")
        f.write(f"- Samples: {metadata['n_samples']}, Features: {metadata['n_features']}\n")
        f.write(f"- Beta: {metadata['beta']}\n")
        f.write(f"- Intercept: {metadata['intercept']}\n")
        f.write("- DGP: eta = X@beta + intercept, p = sigmoid(eta), y ~ Bernoulli(p)\n\n")
        f.write("## 2. Confusion Matrix\n")
        f.write(f"```\n{cm}\n```\n\n")
        f.write("## 3. Model Comparison\n")
        f.write("![Comparison](linear_vs_logistic.png)\n")
        f.write("- Linear Regression outputs unbounded values, cannot be interpreted as probability.\n")
        f.write("- Logistic Regression outputs valid probabilities in [0, 1].\n\n")
        f.write("## 4. Loss Curves\n")
        f.write("![Loss Curves](loss_curves.png)\n")
        f.write("- Log loss heavily penalizes confident mistakes, MSE penalizes linearly.\n")
    
    print(f"📄 Report: {report_path}")
    return report_path


def generate_threshold_report(results_dir, df_threshold):
    report_path = Path(results_dir) / "threshold_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Threshold Analysis Report\n\n")
        f.write("## Formulas\n\n")
        f.write("### Bernoulli Distribution\n")
        f.write("$$Y \\sim Bernoulli(p)$$\n")
        f.write("Y takes value 1 with probability p, 0 with probability 1-p.\n\n")
        f.write("### Likelihood\n")
        f.write("$$L(p; y) = p^y (1-p)^{1-y}$$\n")
        f.write("For a single observation, this is the probability of observing y given p.\n\n")
        f.write("### Negative Log-Likelihood (Log Loss)\n")
        f.write("$$-\\log L(p; y) = -[y\\log p + (1-y)\\log(1-p)]$$\n")
        f.write("Maximizing likelihood is equivalent to minimizing log loss.\n\n")
        
        f.write("## Threshold Scan Results\n")
        f.write("![Threshold Curves](threshold_curves.png)\n\n")
        f.write("| Threshold | Accuracy | Precision | Recall | F1 |\n")
        f.write("|-----------|----------|-----------|--------|-----|\n")
        for _, row in df_threshold.iterrows():
            f.write(f"| {row['threshold']:.1f} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |\n")
        f.write("\n## Business Scenario: Credit Default\n")
        f.write("- Most important: Recall (catch defaulters)\n")
        f.write("- Recommended threshold: 0.3 (lower threshold to catch more positives)\n")
    
    print(f"📄 Report: {report_path}")
    return report_path


def generate_regularization_report(results_dir, results):
    report_path = Path(results_dir) / "regularization_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Regularization Report: L1 vs L2\n\n")
        f.write("## Performance Comparison\n")
        f.write("![Regularization Comparison](regularization_comparison.png)\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log Loss | Non-zero Coefs |\n")
        f.write("|-------|----------|-----------|--------|-----|---------|----------|----------------|\n")
        for name, res in results.items():
            f.write(f"| {name} | {res['accuracy']:.4f} | {res['precision']:.4f} | {res['recall']:.4f} | {res['f1']:.4f} | {res['roc_auc']:.4f} | {res['log_loss']:.4f} | {res['n_nonzero']} |\n")
        f.write("\n## Key Questions\n")
        f.write("1. **L1 vs L2 prediction**: Similar performance, L1 slightly more sparse.\n")
        f.write("2. **More sparse**: L1 (fewer non-zero coefficients).\n")
        f.write("3. **Better for short variable list**: L1.\n")
        f.write("4. **Better for stability**: L2 (keeps all variables, more stable).\n")
    
    print(f"📄 Report: {report_path}")
    return report_path


def generate_summary_report(results_dir):
    summary_path = Path(results_dir) / "summary.md"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Week 15: Summary\n\n")
        f.write("## 1. Logistic Regression is not just Linear Regression + sigmoid\n")
        f.write("The key difference is the probabilistic interpretation and the Bernoulli likelihood.\n\n")
        f.write("## 2. Sigmoid, Bernoulli Likelihood, Log Loss\n")
        f.write("- Sigmoid maps linear output to probability.\n")
        f.write("- Bernoulli likelihood measures probability of observed labels.\n")
        f.write("- Log loss is the negative log of Bernoulli likelihood.\n\n")
        f.write("## 3. Accuracy is not enough\n")
        f.write("Accuracy ignores class imbalance and doesn't distinguish error types.\n\n")
        f.write("## 4. L1 vs L2\n")
        f.write("- L1: sparsity, feature selection.\n")
        f.write("- L2: stability, handles correlation.\n\n")
        f.write("## 5. Logistic Regression as a strong baseline\n")
        f.write("- Outputs interpretable probabilities.\n")
        f.write("- Coefficients show variable direction and magnitude.\n")
        f.write("- Fast, stable, and works with regularization.\n")
    
    print(f"📄 Summary: {summary_path}")
    return summary_path


# ========================== 主流程 ==========================
def main():
    results_dir = Path("src/week15/results")
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print("✅ results folder cleaned and recreated")

    data_dir = Path("src/week15/data")
    ensure_dir(data_dir)

    # Task A-D: Synthetic data
    run_synthetic_task(data_dir, results_dir)

    # Task E: Real data (optional)
    try:
        run_real_data_task(data_dir, results_dir)
    except FileNotFoundError as e:
        print(f"\n⚠️ Task E skipped: {e}")
        print("   Please download Telco Customer Churn data from Kaggle:")
        print("   https://www.kaggle.com/datasets/blastchar/telco-customer-churn")

    # Summary
    generate_summary_report(results_dir)

    print("\n" + "="*70)
    print("🎉 Week 15 Assignment Complete!")
    print("="*70)

if __name__ == "__main__":
    main()