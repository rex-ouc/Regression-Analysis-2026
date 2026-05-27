# 完整可运行：自定义回归求解器 + 低维/高维实验 + 工业库对比
import numpy as np
import time
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error


class AnalyticalSolver:
    """正规方程解析解求解器，使用 np.linalg.solve 保证数值稳定性"""

    def __init__(self):
        self.beta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.beta = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.beta


class GradientDescentSolver:
    """全批量梯度下降求解器，手动实现梯度公式"""

    def __init__(self, learning_rate: float = 0.1, epochs: int = 500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        X_b = np.c_[np.ones((n_samples, 1)), X]
        self.beta = np.zeros(n_features + 1)

        for _ in range(self.epochs):
            y_pred = X_b @ self.beta
            gradients = (2 / n_samples) * X_b.T @ (y_pred - y)
            self.beta -= self.learning_rate * gradients

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.beta


def generate_data(
    n_samples: int, n_features: int, noise_scale: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """生成线性回归模拟数据"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_beta = np.array([1.5] + [0.5] * n_features)
    X_b = np.c_[np.ones((n_samples, 1)), X]
    y = X_b @ true_beta + np.random.randn(n_samples) * noise_scale
    return X, y


def evaluate_model(
    solver, X: np.ndarray, y: np.ndarray, model_name: str
) -> dict[str, float | str]:
    """统一评估模型：训练+预测+计时+计算MSE"""
    start_time = time.time()
    solver.fit(X, y)
    y_pred = solver.predict(X)
    end_time = time.time()

    mse = mean_squared_error(y, y_pred)
    return {
        "name": model_name,
        "time": round(end_time - start_time, 4),
        "mse": round(mse, 4),
    }


def main() -> None:
    """主实验流程"""
    # 实验配置
    N_SAMPLES = 10000
    P_LOW = 10
    P_HIGH = 2000

    # 生成数据
    X_low, y_low = generate_data(N_SAMPLES, P_LOW)
    X_high, y_high = generate_data(N_SAMPLES, P_HIGH)

    # 实验 A：低维数据
    print("=" * 60)
    print("【实验 A：低维数据 N=10000, P=10】")
    ana_low = evaluate_model(AnalyticalSolver(), X_low, y_low, "AnalyticalSolver")
    gd_low = evaluate_model(GradientDescentSolver(), X_low, y_low, "GradientDescentSolver")
    print(f"{ana_low['name']:20} | 耗时: {ana_low['time']:6.4f}s | MSE: {ana_low['mse']}")
    print(f"{gd_low['name']:20} | 耗时: {gd_low['time']:6.4f}s | MSE: {gd_low['mse']}")

    # 实验 B：高维数据
    print("\n" + "=" * 60)
    print("【实验 B：高维数据 N=10000, P=2000】")
    ana_high = evaluate_model(AnalyticalSolver(), X_high, y_high, "AnalyticalSolver")
    gd_high = evaluate_model(GradientDescentSolver(), X_high, y_high, "GradientDescentSolver")
    print(f"{ana_high['name']:20} | 耗时: {ana_high['time']:6.4f}s | MSE: {ana_high['mse']}")
    print(f"{gd_high['name']:20} | 耗时: {gd_high['time']:6.4f}s | MSE: {gd_high['mse']}")

    # 实验 C：工业库对比
    print("\n" + "=" * 60)
    print("【实验 C：工业库对比（高维数据）】")

    # Statsmodels OLS
    start = time.time()
    sm_model = sm.OLS(y_high, sm.add_constant(X_high)).fit()
    sm_mse = mean_squared_error(y_high, sm_model.predict())
    sm_time = round(time.time() - start, 4)
    print(f"{'Statsmodels.OLS':20} | 耗时: {sm_time:6.4f}s | MSE: {round(sm_mse, 4)}")

    # Sklearn LinearRegression
    start = time.time()
    lr_model = LinearRegression().fit(X_high, y_high)
    lr_mse = mean_squared_error(y_high, lr_model.predict(X_high))
    lr_time = round(time.time() - start, 4)
    print(f"{'Sklearn LR':20} | 耗时: {lr_time:6.4f}s | MSE: {round(lr_mse, 4)}")

    # Sklearn SGDRegressor
    start = time.time()
    sgd_model = SGDRegressor(
        learning_rate="constant", eta0=0.1, max_iter=500, random_state=42
    ).fit(X_high, y_high)
    sgd_mse = mean_squared_error(y_high, sgd_model.predict(X_high))
    sgd_time = round(time.time() - start, 4)
    print(f"{'Sklearn SGD':20} | 耗时: {sgd_time:6.4f}s | MSE: {sgd_mse}")

    print("\n✅ 全部实验运行完成！")


if __name__ == "__main__":
    main()