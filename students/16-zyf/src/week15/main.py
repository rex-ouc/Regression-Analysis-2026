# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


DATA_DIR = "src/week15/data"
RESULTS_DIR = "src/week15/results"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ======================
# 1. 生成数据（概率模型）
# ======================
def generate_data(n=500, p=4, seed=42):
    np.random.seed(seed)

    X = np.random.normal(0, 1, (n, p))

    beta = np.array([2.0, -1.5, 0.0, 1.2])

    eta = X @ beta

    prob = 1 / (1 + np.exp(-eta))

    y = np.random.binomial(1, prob)

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    df["y"] = y

    return df


# ======================
# 2. threshold评估
# ======================
def evaluate_thresholds(y_true, prob):

    thresholds = np.arange(0.1, 1.0, 0.1)

    rows = []

    for t in thresholds:

        pred = (prob >= t).astype(int)

        rows.append({
            "threshold": t,
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred),
        })

    return pd.DataFrame(rows)


# ======================
# 3. 主函数
# ======================
def main():

    print("Generating data...")
    df = generate_data()

    path = os.path.join(DATA_DIR, "synthetic_binary.csv")
    df.to_csv(path, index=False)

    X = df.drop(columns=["y"]).values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 模型
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    logit = LogisticRegression(max_iter=2000)
    logit.fit(X_train, y_train)

    # 概率
    lr_prob = np.clip(lr.predict(X_test), 0, 1)
    logit_prob = logit.predict_proba(X_test)[:, 1]

    # 分类
    lr_pred = (lr_prob >= 0.5).astype(int)
    logit_pred = (logit_prob >= 0.5).astype(int)

    print("Linear Accuracy:", accuracy_score(y_test, lr_pred))
    print("Logistic Accuracy:", accuracy_score(y_test, logit_pred))

    print("Confusion Matrix (Logistic):")
    print(confusion_matrix(y_test, logit_pred))

    # threshold分析
    lr_table = evaluate_thresholds(y_test, lr_prob)
    logit_table = evaluate_thresholds(y_test, logit_prob)

    # 画图
    plt.figure()
    plt.plot(lr_table["threshold"], lr_table["f1"], label="Linear F1")
    plt.plot(logit_table["threshold"], logit_table["f1"], label="Logistic F1")
    plt.legend()
    plt.xlabel("threshold")
    plt.ylabel("F1")
    plt.title("Threshold vs F1")
    plt.savefig(os.path.join(RESULTS_DIR, "f1_curve.png"))
    plt.close()

    print("DONE")


if __name__ == "__main__":
    main()