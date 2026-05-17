# Synthetic Data Test Report
Generated: 2026-05-08 08:14:20

## Data Generation Parameters
- Number of samples: 1000
- True intercept: 1.5
- True coefficients: [2.5, -1.0, 0.5]
- Noise standard deviation: 0.5

## Model Comparison
## Model Performance Comparison

| Model | Fit Time (sec) | R² Score | RMSE | MAE |
|-------|----------------|----------|------|-----|
| CustomOLS | 0.03318 | 0.961652 | 0.518469 | 0.413561 |
| sklearn | 0.02618 | 0.961652 | 0.518469 | 0.413561 |

## Coefficient Estimates

### CustomOLS
================================================================================
Custom OLS Regression Results
================================================================================
Dep. Variable:     y
Model:             OLS
No. Observations:  800
Df Residuals:      796
Df Model:          3
Sigma^2:           0.263077
--------------------------------------------------------------------------------
                     coef      std err            t        P>|t|
--------------------------------------------------------------------------------
Intercept        1.489780     0.018214    81.794132     0.000000
X1               2.495403     0.018552   134.510950     0.000000
X2              -1.042127     0.018251   -57.098261     0.000000
X3               0.532155     0.018519    28.735535     0.000000
--------------------------------------------------------------------------------

### sklearn LinearRegression
Intercept: 1.489780
Coefficients: [2.495403361873899, -1.0421274251598243, 0.5321551230174936]

## Key Findings
1. Both models produce identical R² scores, confirming correct implementation.
2. Residual plots show no obvious pattern, suggesting linear model is appropriate.
3. Q-Q plots indicate residuals are approximately normally distributed.
