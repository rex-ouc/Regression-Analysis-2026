import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

def evaluate_sklearn_solvers(X_train, y_train, X_test, y_test):
    """评估工业界API：statsmodels.OLS、sklearn.LinearRegression、SGDRegressor"""
    results = {}
    
    # 1. statsmodels.api.OLS（传统统计）
    start_time = time.time()
    X_train_sm = sm.add_constant(X_train)  # 添加截距
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    train_time_sm = time.time() - start_time
    X_test_sm = sm.add_constant(X_test)
    y_pred_sm = model_sm.predict(X_test_sm)
    mse_sm = mean_squared_error(y_test, y_pred_sm)
    results["statsmodels"] = (train_time_sm, mse_sm)
    
    # 2. sklearn.linear_model.LinearRegression（机器学习解析解）
    start_time = time.time()
    model_sklearn_lr = LinearRegression()
    model_sklearn_lr.fit(X_train, y_train)
    train_time_sklearn_lr = time.time() - start_time
    y_pred_sklearn_lr = model_sklearn_lr.predict(X_test)
    mse_sklearn_lr = mean_squared_error(y_test, y_pred_sklearn_lr)
    results["sklearn_linear_regression"] = (train_time_sklearn_lr, mse_sklearn_lr)
    
    # 3. sklearn.linear_model.SGDRegressor（机器学习梯度下降）
    start_time = time.time()
    model_sgd = SGDRegressor(loss="squared_error", max_iter=1000, tol=1e-3, random_state=42)
    model_sgd.fit(X_train, y_train)
    train_time_sgd = time.time() - start_time
    y_pred_sgd = model_sgd.predict(X_test)
    mse_sgd = mean_squared_error(y_test, y_pred_sgd)
    results["sklearn_sgdregressor"] = (train_time_sgd, mse_sgd)
    
    return results

# 在高维场景下评估工业界API
high_dim_results = evaluate_sklearn_solvers(X_train_high, y_train_high, X_test_high, y_test_high)

# 打印工业界API结果
print("\n=== 高维场景下工业界API表现 ===")
for api_name, (time_val, mse_val) in high_dim_results.items():
    print(f"{api_name}: 时间={time_val:.4f}s, MSE={mse_val:.4f}")