import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_b = np.hstack([np.ones((n_samples, 1)), X])
        I = np.eye(X_b.shape[1])
        I[0, 0] = 0
        self.coef_ = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

class LassoRegression:
    def __init__(self, alpha=0.1, n_iters=1000, lr=0.01):
        self.alpha = alpha
        self.n_iters = n_iters
        self.lr = lr
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for _ in range(self.n_iters):
            y_pred = X @ self.coef_ + self.intercept_
            dw = (1 / n_samples) * (X.T @ (y_pred - y)) + self.alpha * np.sign(self.coef_)
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.coef_ -= self.lr * dw
            self.intercept_ -= self.lr * db

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

def load_data():
    df = pd.read_csv("dirty_marketing.csv")
    df = df.dropna()
    X = df.drop("Sales", axis=1).select_dtypes(include=[np.number])
    y = df["Sales"]
    return X, y

def main():
    X, y = load_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("===== 岭回归（Ridge）=====")
    ridge = RidgeRegression(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    print(f"Ridge MSE: {mse_ridge:.4f}")
    print(f"Ridge R²: {r2_ridge:.4f}\n")

    print("===== Lasso 回归 =====")
    lasso = LassoRegression(alpha=0.1, n_iters=1000, lr=0.01)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)
    print(f"Lasso MSE: {mse_lasso:.4f}")
    print(f"Lasso R²: {r2_lasso:.4f}\n")

    os.makedirs("results", exist_ok=True)
    with open("results/report.md", "w", encoding="utf-8") as f:
        f.write(f"""# Week09 正则化回归报告（23_zy）

## 模型结果
- 岭回归 MSE: {mse_ridge:.4f}
- 岭回归 R²: {r2_ridge:.4f}
- Lasso 回归 MSE: {mse_lasso:.4f}
- Lasso 回归 R²: {r2_lasso:.4f}
""")

    print("✅ 运行完成！报告已保存到 results/report.md")

if __name__ == "__main__":
    main()
