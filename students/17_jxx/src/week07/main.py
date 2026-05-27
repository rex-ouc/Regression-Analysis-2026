import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append(str(Path(__file__).parent.parent))
from utils.models import AnalyticalOLS, GradientDescentOLS

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def task2_cv(X, y):
    print("\n===== Task 2: 5-Fold Cross-Validation =====")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_list = []
    rmse_list = []
    for i, (tr, val) in enumerate(kf.split(X), 1):
        model = AnalyticalOLS().fit(X[tr], y[tr])
        yp = model.predict(X[val])
        r2 = r2_score(y[val], yp)
        rm = rmse(y[val], yp)
        r2_list.append(r2)
        rmse_list.append(rm)
        print(f"Fold {i}: R2={r2:.4f} | RMSE={rm:.4f}")
    print(f"\nMean R2 = {np.mean(r2_list):.4f}")
    print(f"Mean RMSE = {np.mean(rmse_list):.4f}")
    return np.mean(r2_list), np.mean(rmse_list)

def task3_tune(X_train, y_train, X_val, y_val):
    print("\n===== Task 3: Learning Rate Tuning =====")
    lrs = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    best = -1e9
    best_lr = None
    for lr in lrs:
        m = GradientDescentOLS(learning_rate=lr, gd_type="mini_batch").fit(X_train, y_train)
        r2 = r2_score(y_val, m.predict(X_val))
        rm = rmse(y_val, m.predict(X_val))
        print(f"LR={lr:<8} | Val R2={r2:.4f} | RMSE={rm:.4f}")
        if r2 > best:
            best = r2
            best_lr = lr
    print(f"\nBest Learning Rate = {best_lr}")
    return best_lr

def task4_learning_curve(X_train, y_train, save_path):
    print("\n===== Task 4: Plot Learning Curve =====")
    m1 = GradientDescentOLS(learning_rate=0.01, gd_type="full_batch", max_iter=300).fit(X_train, y_train)
    m2 = GradientDescentOLS(learning_rate=0.01, gd_type="mini_batch", max_iter=300).fit(X_train, y_train)
    plt.figure(figsize=(10,5))
    plt.plot(m1.loss_history_, label="Full Batch")
    plt.plot(m2.loss_history_, label="Mini Batch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Full vs Mini Batch GD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "learning_curve.png", dpi=150)
    plt.close()

def main():
    # ✅ 核心修复：文件生成在 YOUR 文件夹里！
    YOUR_FOLDER = Path(__file__).parent.parent.parent  # students/17_jxx
    results_dir = YOUR_FOLDER / "results"
    results_dir.mkdir(exist_ok=True)

    # 自动找数据
    project_root = Path(__file__).parents[4]
    csv = list(project_root.rglob("q3_marketing.csv"))[0]
    df = pd.read_csv(csv)

    y = df.filter(regex="Sale").values.ravel()
    X = df.select_dtypes(include=[np.number]).drop(columns=df.filter(regex="Sale").columns).values

    X_wb = np.c_[np.ones(len(X)), X]
    cv_r2, cv_rmse = task2_cv(X_wb, y)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    X_train_s = np.c_[np.ones(len(X_train_s)), X_train_s]
    X_val_s = np.c_[np.ones(len(X_val_s)), X_val_s]
    X_test_s = np.c_[np.ones(len(X_test_s)), X_test_s]

    best_lr = task3_tune(X_train_s, y_train, X_val_s, y_val)

    gd = GradientDescentOLS(learning_rate=best_lr, gd_type="mini_batch").fit(X_train_s, y_train)
    ols = AnalyticalOLS().fit(X_train_s, y_train)

    gd_r2 = r2_score(y_test, gd.predict(X_test_s))
    gd_rm = rmse(y_test, gd.predict(X_test_s))
    ols_r2 = r2_score(y_test, ols.predict(X_test_s))
    ols_rm = rmse(y_test, ols.predict(X_test_s))

    print("\n===== Final Test Performance =====")
    print(f"GradientDescent | R2={gd_r2:.4f} | RMSE={gd_rm:.4f}")
    print(f"AnalyticalOLS   | R2={ols_r2:.4f} | RMSE={ols_rm:.4f}")

    task4_learning_curve(X_train_s, y_train, results_dir)

    report = f"""# Week07 Report
- 5折CV平均R2: {cv_r2:.4f}
- 最佳学习率: {best_lr}
- 梯度下降测试R2: {gd_r2:.4f}
- 解析解测试R2: {ols_r2:.4f}
- 标准化仅使用训练集，无数据泄露
"""
    (results_dir / "summary_report.md").write_text(report, encoding="utf-8")
    print(f"\n✅ 文件已生成在：{results_dir}")

if __name__ == "__main__":
    main()