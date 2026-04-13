# Week 04 Assignment: The Tale of Two Solvers (求解器双城记)
# Task 2: 低维 vs 高维算力大比拼 (Dimensionality Benchmark)
# Task 3: 工业界 API 终极对决 (Statsmodels vs Scikit-Learn)

import numpy as np
import time
from solvers import AnalyticalSolver, GradientDescentSolver
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, SGDRegressor

def generate_data(N: int, P: int):
    """Generate synthetic data for linear regression."""
    # Generate features
    X = np.random.randn(N, P)
    # Add intercept term
    X = np.column_stack([np.ones(N), X])
    # True coefficients (including intercept)
    true_beta = np.random.randn(P + 1)
    # Generate target with noise
    y = X @ true_beta + 0.1 * np.random.randn(N)
    return X, y, true_beta

def run_experiment():
    """Run the benchmark experiments."""
    results = {}

    experiments = {
        "A": (10000, 10),  # Low-dimensional
        "B": (10000, 2000),  # High-dimensional
    }

    solvers = {
        "Analytical": AnalyticalSolver(),
        "GradientDescent": GradientDescentSolver(learning_rate=0.01, max_iter=1000),
    }

    for exp_name, (N, P) in experiments.items():
        print(f"\nRunning Experiment {exp_name}: N={N}, P={P}")
        X, y, true_beta = generate_data(N, P)

        for solver_name, solver in solvers.items():
            print(f"  Fitting with {solver_name}Solver...")
            solver.fit(X, y)
            y_pred = solver.predict(X)
            mse = np.mean((y_pred - y) ** 2)
            results[f"{exp_name}_{solver_name}"] = {
                "time": solver.fit_time_,
                "mse": mse,
            }
            print(f"    Time: {solver.fit_time_:.4f}s, MSE: {mse:.4f}")

    # Print summary table
    print("\n" + "=" * 70)
    print("Benchmark Results Summary (Custom Solvers)")
    print("=" * 70)
    print(f"{'Experiment':<15} {'Solver':<20} {'Time (s)':<15} {'MSE':<15}")
    print("-" * 70)
    for exp in ["A", "B"]:
        for solver in ["Analytical", "GradientDescent"]:
            key = f"{exp}_{solver}"
            time_val = results[key]["time"]
            mse_val = results[key]["mse"]
            exp_label = f"Exp {exp}"
            print(f"{exp_label:<15} {solver:<20} {time_val:<15.6f} {mse_val:<15.6f}")
    print("=" * 70)

    # Task 3: Industrial API Showdown on High-Dimensional Data
    print("\n" + "=" * 60)
    print("Task 3: Industrial API Showdown (High-Dimensional Data)")
    print("=" * 60)

    # Use the same high-dimensional data from Experiment B
    N, P = 10000, 2000
    X, y, true_beta = generate_data(N, P)
    X_sklearn = X[:, 1:]  # Remove intercept for sklearn (it adds automatically)

    industrial_results = {}

    # Statsmodels OLS
    print("Fitting with statsmodels.api.OLS...")
    start_time = time.perf_counter()
    model_ols = sm.OLS(y, X).fit(disp=False)
    time_ols = time.perf_counter() - start_time
    industrial_results["Statsmodels OLS"] = time_ols
    print(f"  Time: {time_ols:.4f}s")

    # Scikit-Learn LinearRegression
    print("Fitting with sklearn.linear_model.LinearRegression...")
    start_time = time.perf_counter()
    model_lr = LinearRegression()
    model_lr.fit(X_sklearn, y)
    time_lr = time.perf_counter() - start_time
    industrial_results["Sklearn LinearRegression"] = time_lr
    print(f"  Time: {time_lr:.4f}s")

    # Scikit-Learn SGDRegressor
    print("Fitting with sklearn.linear_model.SGDRegressor...")
    start_time = time.perf_counter()
    model_sgd = SGDRegressor(max_iter=1000, random_state=42)
    model_sgd.fit(X_sklearn, y)
    time_sgd = time.perf_counter() - start_time
    industrial_results["Sklearn SGDRegressor"] = time_sgd
    print(f"  Time: {time_sgd:.4f}s")

    # Print industrial results table
    print("\n" + "=" * 60)
    print("Industrial API Timing Results (High-Dimensional: N=10000, P=2000)")
    print("=" * 60)
    print(f"{'API':<35} {'Time (s)':<20}")
    print("-" * 60)
    for name, t in industrial_results.items():
        print(f"{name:<35} {t:<20.6f}")
    print("=" * 60)

    return results, industrial_results

if __name__ == "__main__":
    run_experiment()
