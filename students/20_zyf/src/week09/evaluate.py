"""
Script: evaluate.py
Purpose: Model diagnostics and 5-fold cross-validation evaluation.
Detects multicollinearity and computes baseline model performance.
"""
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os
import subprocess

# Add utils directory to path for imports
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)

from diagnostics import calculate_vif
from models import AnalyticalOLS


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def main():
    """Main evaluation pipeline."""
    
    # Load cleaned data
    input_path = "homework/week09/data/dirty_marketing.csv"
    clean_data_path = "students/20_zyf/results/week09/clean_marketing.csv"
    result_path = "students/20_zyf/results/week09/diagnostics_report.md"
    
    # Create results directory
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    # Open file for writing results
    with open(result_path, 'w', encoding='utf-8') as f_out:
        def print_and_log(msg=""):
            """Print to both console and file."""
            print(msg)
            f_out.write(msg + "\n")
            f_out.flush()
        
        # ============================================================
        # LOAD AND PREPARE DATA
        # ============================================================
        print_and_log(f"# 🛠️ Week 9: Model Diagnostics & Cross-Validation")
        print_and_log(f"")
        print_and_log(f"**Generated:** {pd.Timestamp.now()}")
        print_and_log(f"")
        print_and_log(f"## Data Information")
        print_and_log(f"")
        
        # Check if clean data exists, if not create it
        try:
            df_clean = pd.read_csv(clean_data_path)
            print_and_log(f"[INFO] Loaded cleaned data from: {clean_data_path}")
        except FileNotFoundError:
            print_and_log(f"[INFO] Clean data not found. Running data preprocessing...")
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(clean_data_path), exist_ok=True)
            
            # Run data preprocessing
            result = subprocess.run([
                sys.executable,
                "students/20_zyf/src/week09/data_prep.py",
                "--input", input_path,
                "--output", clean_data_path
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print_and_log(f"[ERROR] Data preprocessing failed:")
                print_and_log(result.stdout)
                print_and_log(result.stderr)
                raise RuntimeError("Data preprocessing failed")
            
            print_and_log(result.stdout)
            df_clean = pd.read_csv(clean_data_path)
        
        print_and_log(f"[INFO] Data shape: {df_clean.shape}")
        print_and_log(f"[INFO] Columns: {list(df_clean.columns)}\n")
        
        # Prepare features and target
        # Sales is the target, all other columns except Sales are features
        target_col = 'Sales'
        feature_cols = [col for col in df_clean.columns if col != target_col]
        
        X = df_clean[feature_cols].values.astype(np.float64)
        y = df_clean[target_col].values.astype(np.float64)
        
        print_and_log(f"- **Shape:** {df_clean.shape[0]} samples × {df_clean.shape[1]} features")
        print_and_log(f"- **Features:** {', '.join(feature_cols)}")
        print_and_log(f"- **Target:** {target_col}")
        print_and_log(f"")
        
        # ============================================================
        # TASK 1: MULTICOLLINEARITY DIAGNOSTICS
        # ============================================================
        print_and_log(f"## Task 1: Multicollinearity Diagnostics")
        print_and_log(f"")
        
        vif_values = calculate_vif(X)
        
        print_and_log(f"| Feature | VIF Value | Status |")
        print_and_log(f"|---------|-----------|--------|")
        
        high_vif_features = []
        for i, (feature, vif) in enumerate(zip(feature_cols, vif_values)):
            if vif > 10:
                status = "🔴 SEVERE"
                high_vif_features.append((feature, vif))
            elif vif > 5:
                status = "🟡 Moderate"
            else:
                status = "🟢 OK"
            
            print_and_log(f"| {feature} | {vif:.4f} | {status} |")
        
        print_and_log(f"")
        if high_vif_features:
            print_and_log(f"### ⚠️ WARNING: HIGH MULTICOLLINEARITY DETECTED!")
            print_and_log(f"")
            print_and_log(f"The following features have **VIF > 10**:")
            print_and_log(f"")
            for feature, vif in high_vif_features:
                print_and_log(f"- **{feature}**: VIF = {vif:.4f}")
            print_and_log(f"")
            print_and_log(f"These features show severe multicollinearity and should be investigated.")
            print_and_log(f"")
        else:
            print_and_log(f"✅ **No severe multicollinearity detected.**")
            print_and_log(f"")
        
        # ============================================================
        # TASK 2: 5-FOLD CROSS-VALIDATION
        # ============================================================
        print_and_log(f"## Task 2: 5-Fold Cross-Validation")
        print_and_log(f"")
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores = []
        
        print_and_log(f"### Cross-Validation Progress")
        print_and_log(f"")
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model = AnalyticalOLS()
            model.fit(X_train, y_train)
            
            # Evaluate on test set
            r2_score = model.score(X_test, y_test)
            r2_scores.append(r2_score)
            
            print_and_log(f"- **Fold {fold_idx}/5:** R² = {r2_score:.6f}")
        
        print_and_log(f"")
        
        # Compute statistics
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        min_r2 = np.min(r2_scores)
        max_r2 = np.max(r2_scores)
        
        print_and_log(f"### Cross-Validation Statistics")
        print_and_log(f"")
        print_and_log(f"| Metric | Value |")
        print_and_log(f"|--------|-------|")
        print_and_log(f"| Mean R² | {mean_r2:.6f} |")
        print_and_log(f"| Std Dev | {std_r2:.6f} |")
        print_and_log(f"| Min R² | {min_r2:.6f} |")
        print_and_log(f"| Max R² | {max_r2:.6f} |")
        print_and_log(f"")
        print_and_log(f"### ⭐ Baseline Cross-Validation Score: {mean_r2:.6f}")
        
        # ============================================================
        # CRITICAL REFLECTION
        # ============================================================
        print_and_log(f"")
        print_and_log(f"## Critical Reflection: Data Leakage Issue")
        print_and_log(f"")
        print_and_log(f"### Question")
        print_and_log(f"")
        print_and_log(f"In `data_prep.py`, we imputed all missing values using the mean of the **ENTIRE** dataset.")
        print_and_log(f"During 5-fold cross-validation, does the test set represent truly \"unseen\" data?")
        print_and_log(f"")
        print_and_log(f"### Answer: **NO!** 🚨")
        print_and_log(f"")
        print_and_log(f"### Why This Is a Problem")
        print_and_log(f"")
        print_and_log(f"The imputation leakage occurs because:")
        print_and_log(f"")
        print_and_log(f"1. **Means Computed on Full Dataset**: We computed statistics on all 1000 samples before splitting")
        print_and_log(f"2. **Information Leakage**: These means implicitly \"leak\" information into the test set")
        print_and_log(f"3. **Contaminated Test Set**: The test set observations are \"contaminated\" with statistics")
        print_and_log(f"   that include their own values indirectly")
        print_and_log(f"")
        print_and_log(f"### Better Approach (Data Leakage Prevention)")
        print_and_log(f"")
        print_and_log(f"```python")
        print_and_log(f"# WRONG (current approach):")
        print_and_log(f"mean = df.mean()  # computed on FULL data")
        print_and_log(f"for fold in cv.splits():")
        print_and_log(f"    df_test = df.iloc[test_idx].fillna(mean)  # test data contaminated!")
        print_and_log(f"")
        print_and_log(f"# RIGHT (proper approach):")
        print_and_log(f"for fold in cv.splits():")
        print_and_log(f"    train_data, test_data = split(df, fold)")
        print_and_log(f"    mean = train_data.mean()  # computed ONLY on train")
        print_and_log(f"    train_data = train_data.fillna(mean)")
        print_and_log(f"    test_data = test_data.fillna(mean)  # apply same mean to test")
        print_and_log(f"```")
        print_and_log(f"")
        print_and_log(f"### Key Takeaway")
        print_and_log(f"")
        print_and_log(f"The **artificially high** $R^2 = 0.9897$ is partly due to this leakage.")
        print_and_log(f"In Week 10, we will implement proper preprocessing that prevents this issue")
        print_and_log(f"by fitting all transformations ONLY on training data.")
        print_and_log(f"")
        print_and_log(f"---")
        print_and_log(f"")
        print_and_log(f"✅ **Report Generation Complete**")


if __name__ == "__main__":
    main()
