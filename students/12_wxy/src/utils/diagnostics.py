import numpy as np
import matplotlib.pyplot as plt

# ====================== VIF 计算（已修复公式！） ======================
def calculate_vif(X):
    n, p = X.shape
    vif = np.zeros(p)
    
    for i in range(p):
        y = X[:, i]
        cols = [j for j in range(p) if j != i]
        x = X[:, cols]
        
        x_b = np.hstack([np.ones((x.shape[0], 1)), x])
        beta = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y
        y_pred = x_b @ beta
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.999
        
        vif[i] = 1 / (1 - r2 + 1e-9)
    
    return vif

def print_vif_report(v, names):
    print("\nVIF 报告:")
    for n, vi in zip(names, v):
        print(f"{n:15} {vi:.1f}")

# ====================== 系数稳定性绘图 ======================
def plot_coef_stability(ols_list, ridge_list, feat_names, path):
    plt.figure(figsize=(10, 5))
    
    ols_mat = np.array(ols_list)
    ridge_mat = np.array(ridge_list)

    plt.boxplot(ols_mat, positions=[1,2,3], widths=0.3, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.boxplot(ridge_mat, positions=[1.4,2.4,3.4], widths=0.3, patch_artist=True, boxprops=dict(facecolor="lightcoral"))

    plt.xticks([1.2, 2.2, 3.2], feat_names)
    plt.title("OLS vs Ridge Coefficient Stability")
    plt.ylabel("Coefficient Value")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def calc_coef_std(coef_list):
    return np.round(np.std(coef_list, axis=0), 4)

# ====================== 矩阵秩、条件数 ======================
def matrix_condition_metrics(X):
    """返回矩阵秩、条件数"""
    rank = np.linalg.matrix_rank(X)
    cond_num = np.linalg.cond(X)
    return {"rank": rank, "condition_number": cond_num}

# ====================== 新增：系数路径图 ======================
def plot_coef_path(coefs_list, feature_names=None, title="Coefficient Path"):
    """
    绘制系数路径图（用于Lasso等正则化方法）
    """
    plt.figure(figsize=(10, 6))
    
    coef_array = np.array(coefs_list)
    
    # 确保形状是 (n_features, n_steps)
    if coef_array.shape[0] > coef_array.shape[1]:
        coef_array = coef_array.T
    
    for i in range(coef_array.shape[0]):
        plt.plot(coef_array[i], linewidth=1, alpha=0.7, 
                label=feature_names[i] if feature_names and i < len(feature_names) else f"Feat_{i}")
    
    plt.xlabel("Regularization Strength (log scale)")
    plt.ylabel("Coefficient Value")
    plt.title(title)
    plt.grid(alpha=0.3)
    
    if feature_names and len(feature_names) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return plt.gcf()

# ====================== 新增：二分类绘图 Week15 ======================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def plot_ols_vs_logistic_single_feature(x_arr, y_arr, ols_pred_arr, lr_prob_arr, save_path):
    """Task A：单特征 OLS vs 逻辑回归输出对比图（输入全部为numpy数组）"""
    fig, ax = plt.subplots(figsize=(10,6))
    sort_idx = np.argsort(x_arr)
    
    ax.scatter(x_arr, y_arr, alpha=0.6, label="True Labels y ∈ {0,1}")
    ax.plot(x_arr[sort_idx], ols_pred_arr[sort_idx], "r-", lw=2, label="OLS Linear Regression")
    ax.plot(x_arr[sort_idx], lr_prob_arr[sort_idx], "g-", lw=2, label="Logistic Regression Sigmoid")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Model Output")
    ax.set_title("OLS vs Logistic Regression Output Comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_mse_logloss_curve(save_path):
    """Task B：MSE与LogLoss损失对比曲线"""
    p = np.linspace(0.01, 0.99, 300)
    from src.utils.metrics import loss_curve_data
    mse1, mse0, log1, log0 = loss_curve_data(p)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(p, mse1, "r--", label="MSE, True Label y=1")
    ax.plot(p, log1, "r-", label="LogLoss, True Label y=1")
    ax.plot(p, mse0, "b--", label="MSE, True Label y=0")
    ax.plot(p, log0, "b-", label="LogLoss, True Label y=0")
    ax.set_xlabel("Predicted Probability p")
    ax.set_ylabel("Loss")
    ax.set_title("MSE vs LogLoss Comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_threshold_metric_tradeoff(df_scan, save_path):
    """Task C：阈值变化下Acc/Precision/Recall/F1曲线"""
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(df_scan["Threshold"], df_scan["Accuracy"], lw=2, label="Accuracy")
    ax.plot(df_scan["Threshold"], df_scan["Precision"], lw=2, label="Precision")
    ax.plot(df_scan["Threshold"], df_scan["Recall"], lw=2, label="Recall")
    ax.plot(df_scan["Threshold"], df_scan["F1"], lw=2, label="F1 Score")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Precision / Recall / Accuracy / F1 Trade-off")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_l1_l2_coef_compare(l1_coefs, l2_coefs, save_path):
    """Task D：L1/L2逻辑回归系数稀疏对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(range(len(l1_coefs)), l1_coefs, color="#2E86AB")
    ax1.set_title("L1 Regularization Coefficients")  # 改为英文
    ax1.set_xlabel("Feature Index")
    ax1.set_ylabel("Coefficient Value")
    ax2.bar(range(len(l2_coefs)), l2_coefs, color="#A23B72")
    ax2.set_title("L2 Regularization Coefficients")
    ax2.set_xlabel("Feature Index")
    ax2.set_ylabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()