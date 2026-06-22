from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, log_loss
)

# 100% 复用你自己写的utils组件
from utils.transformers import CustomStandardScaler

# 全局绘图样式
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# ==================== 工具函数 ====================
def init_results_dir(results_dir: Path, figures_dir: Path):
    """自动清理并初始化结果目录"""
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    print(f"📁 结果目录已初始化: {results_dir}")
    print(f"📁 图表目录已初始化: {figures_dir}")

# ==================== 数据生成函数 ====================
def generate_binary_data(n_samples: int = 400, random_state: int = 42):
    """
    生成带明确概率结构的二分类模拟数据
    DGP: η = Xβ → p = sigmoid(η) → y ~ Bernoulli(p)
    """
    np.random.seed(random_state)
    X = np.random.normal(0, 1, (n_samples, 4))
    # 真实系数：x1强正影响，x2强负影响，x3弱影响，x4纯噪声
    true_beta = np.array([1.5, -1.0, 0.5, 0.0])
    eta = X @ true_beta
    p = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, p)
    feature_names = [f"x{i+1}" for i in range(4)]
    return X, y, feature_names, true_beta

def generate_highdim_binary_data(n_samples: int = 400, n_features: int = 20, random_state: int = 42):
    """生成带共线性和噪声的高维二分类数据"""
    np.random.seed(random_state)
    # 3个潜在因子生成10个相关特征
    Z = np.random.normal(0, 1, (n_samples, 3))
    W = np.random.normal(0, 1, (3, 10))
    X_corr = Z @ W + np.random.normal(0, 0.1, (n_samples, 10))
    # 10个纯噪声特征
    X_noise = np.random.normal(0, 1, (n_samples, 10))
    X = np.hstack([X_corr, X_noise])
    
    # 真实系数：前3个特征有真实影响，其余为噪声
    beta_true = np.zeros(n_features)
    beta_true[:3] = np.array([1.2, -0.8, 0.5])
    eta = X @ beta_true
    p = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, p)
    feature_names = [f"x{i+1}" for i in range(n_features)]
    return X, y, feature_names

# ==================== Task A: 线性回归 vs 逻辑回归 ====================
def run_linear_vs_logistic(X_train, X_test, y_train, y_test, figures_dir: Path):
    """对比线性回归与逻辑回归的输出行为差异"""
    print("\n[Stage 1] 运行线性回归 vs 逻辑回归对比...")
    
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. 线性回归
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    
    # 2. 无正则逻辑回归
    logreg = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    y_prob_logreg = logreg.predict_proba(X_test_scaled)[:, 1]
    
    # 绘图：以x1为横轴展示输出差异
    sort_idx = np.argsort(X_test_scaled[:, 0])
    x1_sorted = X_test_scaled[sort_idx, 0]
    lr_sorted = y_pred_lr[sort_idx]
    logreg_sorted = y_prob_logreg[sort_idx]
    
    plt.figure(figsize=(12, 8))
    plt.scatter(X_test_scaled[:, 0], y_test, alpha=0.5, c="gray", label="True Labels (0/1)")
    plt.plot(x1_sorted, lr_sorted, "b-", linewidth=2, label="Linear Regression Output")
    plt.plot(x1_sorted, logreg_sorted, "r-", linewidth=2, label="Logistic Regression Probability")
    plt.axhline(y=0, color="k", linestyle="--", linewidth=1)
    plt.axhline(y=1, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Feature x1 (Standardized)")
    plt.ylabel("Model Output")
    plt.title("Linear Regression vs Logistic Regression Output Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "linear_vs_logistic.png")
    plt.close()
    
    # 计算准确率
    y_pred_lr_class = (y_pred_lr >= 0.5).astype(int)
    lr_acc = accuracy_score(y_test, y_pred_lr_class)
    logreg_acc = accuracy_score(y_test, logreg.predict(X_test_scaled))
    
    print("✅ 线性回归vs逻辑回归对比图已生成")
    print(f"   线性回归测试准确率: {lr_acc:.4f}")
    print(f"   逻辑回归测试准确率: {logreg_acc:.4f}")
    
    return lr, logreg, X_test_scaled, y_pred_lr, y_prob_logreg

# ==================== Task B: 损失函数对比 ====================
def run_loss_comparison(figures_dir: Path):
    """对比Log Loss与平方误差在二分类任务中的行为"""
    print("\n[Stage 2] 运行损失函数对比...")
    
    p = np.linspace(0.001, 0.999, 1000)
    
    # y=1时的损失
    log_loss_y1 = -np.log(p)
    mse_y1 = (1 - p) ** 2
    
    # y=0时的损失
    log_loss_y0 = -np.log(1 - p)
    mse_y0 = p ** 2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.plot(p, log_loss_y1, "r-", linewidth=2, label="Log Loss")
    ax1.plot(p, mse_y1, "b-", linewidth=2, label="Squared Error")
    ax1.set_xlabel("Predicted Probability p")
    ax1.set_ylabel("Loss Value")
    ax1.set_title("Loss when True Label y = 1")
    ax1.legend()
    
    ax2.plot(p, log_loss_y0, "r-", linewidth=2, label="Log Loss")
    ax2.plot(p, mse_y0, "b-", linewidth=2, label="Squared Error")
    ax2.set_xlabel("Predicted Probability p")
    ax2.set_ylabel("Loss Value")
    ax2.set_title("Loss when True Label y = 0")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(figures_dir / "loss_comparison.png")
    plt.close()
    
    print("✅ 损失函数对比图已生成")

# ==================== Task C: 阈值扫描与指标分析 ====================
def run_threshold_analysis(y_test, y_prob, figures_dir: Path):
    """扫描不同分类阈值，计算各项分类指标"""
    print("\n[Stage 3] 运行阈值扫描分析...")
    
    thresholds = np.arange(0.1, 1.0, 0.1)
    metrics_list = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        metrics_list.append({
            "threshold": round(thresh, 1),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        })
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # 绘制阈值曲线
    plt.figure(figsize=(12, 8))
    plt.plot(metrics_df["threshold"], metrics_df["accuracy"], "o-", label="Accuracy")
    plt.plot(metrics_df["threshold"], metrics_df["precision"], "s-", label="Precision")
    plt.plot(metrics_df["threshold"], metrics_df["recall"], "^-", label="Recall")
    plt.plot(metrics_df["threshold"], metrics_df["f1"], "d-", label="F1")
    plt.xlabel("Classification Threshold")
    plt.ylabel("Metric Value")
    plt.title("Metrics vs Classification Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "threshold_curve.png")
    plt.close()
    
    # 阈值0.5的混淆矩阵
    y_pred_default = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_default).ravel()
    
    print("✅ 阈值分析完成，曲线已生成")
    print(f"   阈值0.5时 Accuracy: {accuracy_score(y_test, y_pred_default):.4f}, F1: {f1_score(y_test, y_pred_default):.4f}")
    
    return metrics_df, tp, tn, fp, fn

# ==================== Task D: L1 vs L2正则化对比 ====================
def run_regularization_comparison(X_train, X_test, y_train, y_test, figures_dir: Path):
    """高维场景下对比L1和L2正则化逻辑回归"""
    print("\n[Stage 4] 运行L1 vs L2正则化对比...")
    
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # L1正则化 + 交叉验证选参
    l1_grid = GridSearchCV(
        LogisticRegression(penalty="l1", solver="liblinear", max_iter=10000),
        param_grid={"C": np.logspace(-3, 2, 20)},
        cv=5, scoring="roc_auc", n_jobs=-1
    )
    l1_grid.fit(X_train_scaled, y_train)
    
    # L2正则化 + 交叉验证选参
    l2_grid = GridSearchCV(
        LogisticRegression(penalty="l2", solver="lbfgs", max_iter=10000),
        param_grid={"C": np.logspace(-3, 2, 20)},
        cv=5, scoring="roc_auc", n_jobs=-1
    )
    l2_grid.fit(X_train_scaled, y_train)
    
    # 测试集评估
    y_prob_l1 = l1_grid.predict_proba(X_test_scaled)[:, 1]
    y_prob_l2 = l2_grid.predict_proba(X_test_scaled)[:, 1]
    y_pred_l1 = l1_grid.predict(X_test_scaled)
    y_pred_l2 = l2_grid.predict(X_test_scaled)
    
    results = {
        "L1": {
            "best_C": l1_grid.best_params_["C"],
            "accuracy": accuracy_score(y_test, y_pred_l1),
            "recall": recall_score(y_test, y_pred_l1),
            "roc_auc": roc_auc_score(y_test, y_prob_l1),
            "log_loss": log_loss(y_test, y_prob_l1),
            "nonzero": int(np.sum(np.abs(l1_grid.best_estimator_.coef_[0]) > 1e-6))
        },
        "L2": {
            "best_C": l2_grid.best_params_["C"],
            "accuracy": accuracy_score(y_test, y_pred_l2),
            "recall": recall_score(y_test, y_pred_l2),
            "roc_auc": roc_auc_score(y_test, y_prob_l2),
            "log_loss": log_loss(y_test, y_prob_l2),
            "nonzero": int(np.sum(np.abs(l2_grid.best_estimator_.coef_[0]) > 1e-6))
        }
    }
    
    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 性能指标对比
    metric_names = ["Accuracy", "Recall", "ROC-AUC"]
    l1_vals = [results["L1"]["accuracy"], results["L1"]["recall"], results["L1"]["roc_auc"]]
    l2_vals = [results["L2"]["accuracy"], results["L2"]["recall"], results["L2"]["roc_auc"]]
    
    x = np.arange(len(metric_names))
    width = 0.35
    ax1.bar(x - width/2, l1_vals, width, label="L1", color="#ff6b6b", alpha=0.7)
    ax1.bar(x + width/2, l2_vals, width, label="L2", color="#4ecdc4", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names)
    ax1.set_ylabel("Metric Value")
    ax1.set_title("Performance Comparison")
    ax1.legend()
    ax1.set_ylim(0.5, 1.0)
    
    # 稀疏性对比
    ax2.bar(["L1", "L2"], [results["L1"]["nonzero"], results["L2"]["nonzero"]],
            color=["#ff6b6b", "#4ecdc4"], alpha=0.7)
    ax2.set_ylabel("Number of Non-zero Coefficients")
    ax2.set_title("Model Sparsity")
    
    plt.tight_layout()
    plt.savefig(figures_dir / "regularization_comparison.png")
    plt.close()
    
    print("✅ L1 vs L2正则化对比完成")
    print(f"   L1最优C={results['L1']['best_C']:.4f}, 非零系数={results['L1']['nonzero']}")
    print(f"   L2最优C={results['L2']['best_C']:.4f}, 非零系数={results['L2']['nonzero']}")
    
    return results

# ==================== 生成报告函数 ====================
def write_synthetic_report(results_dir, X_test_scaled, y_test, lr, logreg, tp, tn, fp, fn):
    """生成模拟数据分析报告"""
    print("\n[Stage 5] 生成模拟数据分析报告...")
    
    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_lr_class = (y_pred_lr >= 0.5).astype(int)
    lr_acc = accuracy_score(y_test, y_pred_lr_class)
    logreg_acc = accuracy_score(y_test, logreg.predict(X_test_scaled))
    
    report_content = """# 二分类模拟数据分析报告

## 1. 数据生成机制(DGP)
### 基本信息
- 总样本量: 400
- 特征数: 4
- 训练集: 280
- 测试集: 120

### 生成过程
1. 生成4个独立标准正态特征 x1 ~ x4
2. 构造线性预测项: η = 1.5·x1 - 1.0·x2 + 0.5·x3 + 0·x4
3. Sigmoid映射为概率: p = 1 / (1 + exp(-η))
4. 从伯努利分布 Bernoulli(p) 采样得到标签 y

### 特征影响
- x1: 正相关，取值越大，正类概率越高
- x2: 负相关，取值越大，正类概率越低
- x3: 弱正相关，对概率影响较小
- x4: 纯噪声，对结果无真实影响

## 2. 线性回归 vs 逻辑回归对比
### 基础表现
- 线性回归测试准确率: {lr_acc:.4f}
- 逻辑回归测试准确率: {logreg_acc:.4f}

### 输出行为差异
1. **取值范围**: 线性回归输出无界，可以小于0或大于1；逻辑回归输出严格限制在(0,1)区间，天然符合概率定义
2. **极端值表现**: 特征极端取值时，线性回归输出会超出合理范围；逻辑回归会平滑趋近于0或1
3. **概率意义**: 线性回归输出没有概率解释；逻辑回归基于极大似然推导，输出可直接解释为事件发生概率

### 核心问题回答
1. **线性回归最不自然的地方是什么？**
   它假设因变量服从正态分布，但二分类标签只有0/1，误差服从二项分布，不满足模型假设。同时输出无上下界，无法被合理解释为概率。

2. **为什么逻辑回归输出更容易解释为概率？**
   它基于伯努利分布建模，通过Sigmoid函数将线性项映射到(0,1)区间，输出天然满足概率的取值要求，有完整的统计学理论支撑。

3. **关键区别是能不能分类还是输出有没有概率意义？**
   关键在于输出是否有概率意义。线性回归也能通过阈值完成分类，但无法给出置信度量化评估；逻辑回归不仅能分类，还能输出校准良好的概率，业务价值更高。

## 3. 混淆矩阵与基础指标(阈值=0.5)
| 指标 | 数值 | 说明 |
|------|------|------|
| TP 真正例 | {tp} | 预测为正，实际为正 |
| TN 真负例 | {tn} | 预测为负，实际为负 |
| FP 假正例 | {fp} | 预测为正，实际为负 |
| FN 假负例 | {fn} | 预测为负，实际为正 |
| Accuracy | {acc:.4f} | 整体正确率 |
| Precision | {prec:.4f} | 预测正例中的真实正例比例 |
| Recall | {rec:.4f} | 真实正例中被识别的比例 |
| F1 | {f1:.4f} | 精确率与召回率的调和平均 |
""".format(
        lr_acc=lr_acc,
        logreg_acc=logreg_acc,
        tp=tp, tn=tn, fp=fp, fn=fn,
        acc=logreg_acc,
        prec=precision_score(y_test, logreg.predict(X_test_scaled)),
        rec=recall_score(y_test, logreg.predict(X_test_scaled)),
        f1=f1_score(y_test, logreg.predict(X_test_scaled))
    )
    
    with open(results_dir / "synthetic_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("✅ 模拟数据分析报告已生成")

def write_threshold_report(results_dir, metrics_df):
    """生成阈值与损失函数分析报告"""
    print("\n[Stage 6] 生成阈值分析报告...")
    
    # 使用原始字符串避免LaTeX转义问题
    report_content = r"""# 阈值与损失函数分析报告

## 1. 核心公式与解释
### 1.1 伯努利分布
$$Y \sim Bernoulli(p)$$
描述单次二项试验的概率分布，随机变量只能取0或1。取1的概率为p，取0的概率为1-p，是二分类问题的基础概率模型。

### 1.2 单样本似然函数
$$L(p;y) = p^y (1-p)^{1-y}$$
表示在参数p下，观测到样本y的概率。y=1时似然为p，y=0时似然为1-p。极大似然估计的目标就是找到p让观测到所有样本的联合概率最大。

### 1.3 负对数似然(Log Loss)
$$-\log L(p;y) = -\left[ y\log p + (1-y)\log(1-p) \right]$$
最大化似然等价于最小化负对数似然，也就是Log Loss。取对数将乘法转化为加法，既避免数值下溢，也让优化问题更简单。

## 2. 损失函数对比分析
我们对比了Log Loss和平方误差在真实标签固定时的变化规律：

### 核心观察
1. **错得很自信时的惩罚**: Log Loss在预测完全错误且置信度极高时（如y=1但p→0），损失会趋向无穷大，惩罚极重；而平方误差最多为1，惩罚力度有限。
2. **梯度特性**: Log Loss的梯度随错误程度增大而增大，能给模型更强的修正信号；平方误差在极端区域梯度会变小，学习效率低。
3. **概率校准**: 最小化Log Loss等价于拟合真实概率，能保证输出概率的校准性；平方误差没有这个性质。

### 为什么Log Loss更自然
1. 它不是人为指定的损失，而是直接从伯努利分布的极大似然推导而来
2. 它天然适配概率输出，鼓励模型输出校准良好的概率值
3. 它对"自信地猜错"施加重罚，符合分类任务的直觉

## 3. 阈值扫描分析
我们扫描了0.1到0.9的分类阈值，观察各项指标变化：

| 阈值 | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
"""
    
    for _, row in metrics_df.iterrows():
        report_content += f"| {row['threshold']:.1f} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |\n"
    
    report_content += """
### 指标变化规律
- **Precision**: 阈值升高而上升。阈值越高，模型只对高置信度样本判正，预测正例的纯度越高。
- **Recall**: 阈值升高而下降。阈值越高，越多正类样本被漏检，召回率越低。
- **Accuracy与F1**: 通常在中间阈值达到峰值，呈现先升后降的趋势。

### 业务场景解读：疾病初筛
1. **最在意的指标**: Recall（召回率）。
2. **原因**: 初筛的核心目标是不漏掉真正的患者。假阳性可以通过后续检查排除，但假阴性会延误治疗，代价极高。
3. **阈值选择**: 会选择较低的阈值（如0.3），牺牲精确率换取高召回率，最大化疾病检出率。
"""
    
    with open(results_dir / "threshold_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("✅ 阈值分析报告已生成")

def write_regularization_report(results_dir, reg_results):
    """生成正则化对比报告"""
    print("\n[Stage 7] 生成正则化对比报告...")
    
    report_content = """# L1 vs L2正则化逻辑回归对比报告

## 1. 实验设置
- 数据集: 20维高维二分类数据，含10个相关特征+10个噪声特征
- 预处理: 全部特征标准化
- 超参数选择: 5折交叉验证，以ROC-AUC为指标选最优C
- 对比对象: L1正则化逻辑回归、L2正则化逻辑回归

## 2. 结果对比
| 指标 | L1正则化 | L2正则化 |
|------|----------|----------|
| 最优正则化参数C | {l1_C:.4f} | {l2_C:.4f} |
| 测试集Accuracy | {l1_acc:.4f} | {l2_acc:.4f} |
| 测试集Recall | {l1_rec:.4f} | {l2_rec:.4f} |
| 测试集ROC-AUC | {l1_auc:.4f} | {l2_auc:.4f} |
| 测试集Log Loss | {l1_logloss:.4f} | {l2_logloss:.4f} |
| 非零系数个数 | {l1_nonzero} | {l2_nonzero} |

## 3. 核心结论
1. **预测性能**: 两者预测表现差异不大，都能有效抑制过拟合，泛化能力接近。
2. **稀疏性**: L1模型明显更稀疏，会将噪声特征的系数完全压缩为0，实现自动变量筛选；L2只会收缩系数，不会剔除任何特征。
3. **适用场景**:
   - 需要精简变量名单 → 选L1，输出可解释性强的特征子集
   - 看重模型稳定性 → 选L2，解更平滑，对数据波动不敏感
""".format(
        l1_C=reg_results["L1"]["best_C"],
        l2_C=reg_results["L2"]["best_C"],
        l1_acc=reg_results["L1"]["accuracy"],
        l2_acc=reg_results["L2"]["accuracy"],
        l1_rec=reg_results["L1"]["recall"],
        l2_rec=reg_results["L2"]["recall"],
        l1_auc=reg_results["L1"]["roc_auc"],
        l2_auc=reg_results["L2"]["roc_auc"],
        l1_logloss=reg_results["L1"]["log_loss"],
        l2_logloss=reg_results["L2"]["log_loss"],
        l1_nonzero=reg_results["L1"]["nonzero"],
        l2_nonzero=reg_results["L2"]["nonzero"]
    )
    
    with open(results_dir / "regularization_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("✅ 正则化对比报告已生成")

def write_summary(results_dir):
    """生成总结报告"""
    print("\n[Stage 8] 生成总结报告...")
    
    report_content = """# 逻辑回归与二分类总结报告

## 1. 为什么逻辑回归不是"线性回归接Sigmoid"这么简单？
表面上只是多了一个非线性变换，但本质上两者建模假设完全不同：
- 线性回归假设因变量服从正态分布，最小化平方误差，目标是预测连续值期望
- 逻辑回归假设因变量服从伯努利分布，最大化对数似然，目标是建模事件概率

Sigmoid不是随意选择的映射函数，它是伯努利分布指数族形式的自然结果。逻辑回归有完整的概率统计基础，不是简单的"非线性线性回归"。

## 2. Sigmoid、Bernoulli似然、Log Loss的关系
三者是统一的整体，层层递进：
1. Bernoulli分布是二分类的基础概率假设，描述标签的生成机制
2. Sigmoid是伯努利分布的规范链接函数，将线性项映射到概率区间
3. Log Loss是伯努利分布的负对数似然，最小化它等价于最大化观测数据的似然

三者共同构成了逻辑回归的理论基石，不是三个独立的组件。

## 3. 为什么分类模型不能只看Accuracy？
Accuracy有严重的局限性：
1. **类别不平衡失效**: 正负样本悬殊时，全猜多数类就能得到很高的准确率，但完全没有识别能力
2. **错误代价无差异**: 它不区分假阳性和假阴性，而实际业务中两类错误的代价往往天差地别
3. **丢失置信信息**: 只看最终分类结果，不关心模型的把握程度，损失大量有价值信息

实际业务中必须根据场景选择Precision、Recall、ROC-AUC等更合适的指标。

## 4. L1和L2逻辑回归分别适合什么目标？
- **L1正则化**: 适合变量筛选与模型简化。能产生稀疏解，自动剔除不重要特征，给出精简的变量名单。当业务需要明确"哪些因素重要"时首选。
- **L2正则化**: 适合追求稳定性与泛化能力。解更平滑稳定，对数据波动不敏感。存在高度相关特征、更看重概率可靠性时更优。

## 5. 为什么逻辑回归仍是很强的Baseline？
如果业务需要"输出稳定概率+可解释变量方向"的模型，逻辑回归依然是首选：
1. **可解释性强**: 系数有明确的统计意义，能直接说明特征的影响方向和程度
2. **概率校准好**: 基于极大似然估计，输出概率天然校准，无需额外处理
3. **稳定性高**: 尤其是L2正则化版本，模型行为可预测，不易出意外
4. **计算高效**: 训练预测速度快，易于部署和迭代
5. **理论成熟**: 有坚实的统计基础，是工业界长期验证的基线模型
"""
    
    with open(results_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("✅ 总结报告已生成")

# ==================== 主函数 ====================
def main():
    base_dir = Path(__file__).parent
    data_path = base_dir / "data" / "synthetic_binary.csv"
    results_dir = base_dir / "results"
    figures_dir = results_dir / "figures"
    
    # 初始化目录
    init_results_dir(results_dir, figures_dir)
    
    # ========== 基础二分类数据实验 ==========
    X, y, feature_names, _ = generate_binary_data()
    df = pd.DataFrame(np.column_stack([X, y]), columns=feature_names + ["y"])
    df.to_csv(data_path, index=False)
    print(f"✅ 二分类模拟数据已保存到: {data_path}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Stage1-3: 基础对比、损失函数、阈值分析
    lr, logreg, X_test_scaled, _, y_prob = run_linear_vs_logistic(
        X_train, X_test, y_train, y_test, figures_dir
    )
    run_loss_comparison(figures_dir)
    metrics_df, tp, tn, fp, fn = run_threshold_analysis(y_test, y_prob, figures_dir)
    
    # ========== 高维正则化实验 ==========
    X_high, y_high, _ = generate_highdim_binary_data()
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_high, y_high, test_size=0.3, random_state=42
    )
    reg_results = run_regularization_comparison(
        X_train_h, X_test_h, y_train_h, y_test_h, figures_dir
    )
    
    # ========== 生成所有报告 ==========
    write_synthetic_report(results_dir, X_test_scaled, y_test, lr, logreg, tp, tn, fp, fn)
    write_threshold_report(results_dir, metrics_df)
    write_regularization_report(results_dir, reg_results)
    write_summary(results_dir)
    
    print("\n" + "="*50)
    print("🎉 所有任务完成！图表和报告已保存到 results/ 文件夹")
    print("="*50)

if __name__ == "__main__":
    main()