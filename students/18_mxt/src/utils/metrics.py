import numpy as np


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, skipping near-zero denominators."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > 1e-6
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator != 0 else 0.0


def classification_counts(y_true, y_pred) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "TP": int(np.sum((y_true == 1) & (y_pred == 1))),
        "TN": int(np.sum((y_true == 0) & (y_pred == 0))),
        "FP": int(np.sum((y_true == 0) & (y_pred == 1))),
        "FN": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def classification_metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def classification_metrics_at_threshold(y_true, positive_proba, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.asarray(positive_proba) >= threshold).astype(int)
    counts = classification_counts(y_true, y_pred)
    metrics = classification_metrics_from_counts(counts)
    return {**counts, **metrics}


def binary_log_loss(y_true, positive_proba, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(positive_proba).astype(float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
