import numpy as np

def rmse(y_true, y_pred):
    """均方根误差"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred):
    """决定系数"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# ====================== 新增：二分类评估指标 Week15 ======================
def log_loss(y_true, y_pred_prob):
    """负对数似然损失(log loss)，增加极小值防止log(0)溢出"""
    eps = 1e-8
    p = np.clip(y_pred_prob, eps, 1 - eps)
    loss = - np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return loss


def calc_binary_metrics(y_true, y_pred_prob, threshold=0.5):
    """计算混淆矩阵全套指标 TP/TN/FP/FN/Acc/Prec/Recall/F1/AUC/LogLoss"""
    from sklearn.metrics import confusion_matrix, roc_auc_score
    y_pred = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 1e-9 else 0.0
    auc = roc_auc_score(y_true, y_pred_prob)
    ll = log_loss(y_true, y_pred_prob)

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": acc, "Precision": precision,
        "Recall": recall, "F1": f1, "AUC": auc, "LogLoss": ll
    }


def scan_all_thresholds(y_true, y_pred_prob, start=0.1, end=0.9, step=0.1):
    """遍历阈值，批量生成指标表，返回DataFrame"""
    import pandas as pd
    thresholds = np.arange(start, end + step, step)
    records = []
    for t in thresholds:
        met = calc_binary_metrics(y_true, y_pred_prob, threshold=t)
        met["Threshold"] = t
        records.append(met)
    return pd.DataFrame(records)


def loss_curve_data(p_range):
    """生成MSE与LogLoss对比曲线数据 Task B"""
    eps = 1e-8
    p = np.clip(p_range, eps, 1 - eps)
    # MSE loss
    mse_y1 = (1 - p) ** 2
    mse_y0 = (0 - p) ** 2
    # Log loss
    log_y1 = -np.log(p)
    log_y0 = -np.log(1 - p)
    return mse_y1, mse_y0, log_y1, log_y0