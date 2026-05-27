"""
Script: data_prep.py
Purpose: CLI tool for data preprocessing and cleaning.
Handles missing values, outliers (Winsorization), and categorical encoding.
"""
import sys
import argparse
import pandas as pd
import numpy as np


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Data Preprocessing CLI: Clean dirty marketing data"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file (dirty data)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file (clean data)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"[INFO] Loading data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"[INFO] Data shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # Display missing values info
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"[INFO] Missing values detected:\n{missing_counts[missing_counts > 0]}")
    
    # Stage 1: Handle missing values
    print("[STAGE 1] Handling missing values with mean imputation...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            fill_value = df[col].mean()
            df[col].fillna(fill_value, inplace=True)
            print(f"  - Imputed {col} with mean: {fill_value:.4f}")
    
    # Stage 2: Handle outliers (Winsorization)
    print("[STAGE 2] Applying Winsorization at 99th percentile...")
    for col in numeric_cols:
        percentile_99 = df[col].quantile(0.99)
        outlier_count = (df[col] > percentile_99).sum()
        if outlier_count > 0:
            df[col] = df[col].clip(upper=percentile_99)
            print(f"  - Winsorized {col}: {outlier_count} outliers capped at {percentile_99:.4f}")
    
    # Stage 3: Handle categorical variables (One-Hot encoding with drop_first)
    print("[STAGE 3] Processing categorical variables with One-Hot encoding...")
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        # Identify columns to encode
        cols_to_drop = []
        for col in categorical_cols:
            print(f"  - One-Hot encoding '{col}' (drop_first=True)...")
            # Use pd.get_dummies with drop_first=True to avoid dummy variable trap
            encoded = pd.get_dummies(df[[col]], prefix=col, drop_first=True)
            cols_to_drop.append(col)
            df = pd.concat([df, encoded], axis=1)
        
        # Drop original categorical columns
        df = df.drop(columns=cols_to_drop)
        print(f"  - Removed original categorical columns: {cols_to_drop}")
    
    print(f"[INFO] Final data shape: {df.shape}")
    print(f"[INFO] Final columns: {list(df.columns)}")
    
    # Save cleaned data
    print(f"[INFO] Saving cleaned data to: {args.output}")
    df.to_csv(args.output, index=False)
    print("[SUCCESS] Data preprocessing completed!")
    print(f"[SUMMARY] Processed {df.shape[0]} rows × {df.shape[1]} columns")


if __name__ == "__main__":
    main()
