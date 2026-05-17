# Real-World Marketing Data Analysis Report
Generated: 2026-05-08 08:14:24

## Hypothesis Tests (F-Tests)

### Test 1: Joint Significance of Advertising (TV + Radio + Social)
H₀: β_TV = β_Radio = β_Social = 0

| Market | F-statistic | p-value | Reject H₀? | Conclusion |
|--------|-------------|---------|------------|------------|
| NA | 21.6115 | 0.000000 | ✓ Yes | Advertising has significant effect |
| EU | 14.1251 | 0.000000 | ✓ Yes | Advertising has significant effect |

### Test 2: Holiday Promotion Effect
H₀: β_Holiday = 0

| Market | F-statistic | p-value | Reject H₀? | Conclusion |
|--------|-------------|---------|------------|------------|
| NA | 65.7251 | 0.000000 | ✓ Yes | Holiday promotions affect sales |
| EU | 0.8961 | 0.347334 | ✗ No | Holiday promotions have no effect |

## Detailed Model Summaries

### NA Market
```
================================================================================
Custom OLS Regression Results
================================================================================
Dep. Variable:     y
Model:             OLS
No. Observations:  70
Df Residuals:      65
Df Model:          4
Sigma^2:           86.511376
--------------------------------------------------------------------------------
                     coef      std err            t        P>|t|
--------------------------------------------------------------------------------
Intercept       98.791282     3.461070    28.543568     0.000000
X1               0.052604     0.006581     7.993054     0.000000
X2               0.006554     0.013688     0.478864     0.633643
X3               0.027636     0.020106     1.374516     0.174002
X4              18.541682     2.287092     8.107100     0.000000
--------------------------------------------------------------------------------
```

### EU Market
```
================================================================================
Custom OLS Regression Results
================================================================================
Dep. Variable:     y
Model:             OLS
No. Observations:  70
Df Residuals:      65
Df Model:          4
Sigma^2:           126.653028
--------------------------------------------------------------------------------
                     coef      std err            t        P>|t|
--------------------------------------------------------------------------------
Intercept       79.222182     4.088330    19.377640     0.000000
X1               0.048852     0.009513     5.135056     0.000003
X2               0.051216     0.015485     3.307430     0.001537
X3               0.004860     0.023686     0.205167     0.838083
X4               2.986019     3.154373     0.946628     0.347334
--------------------------------------------------------------------------------
```

## Business Insights

### Key Findings

**NA Market:**
- Most effective channel: TV
- Holiday effect: $18.54 per unit

**EU Market:**
- Most effective channel: Radio
- Holiday effect: $2.99 per unit

## OOP Advantages Demonstrated

1. **Multiple Independent Instances**: Each market has its own `CustomOLS` instance
2. **Encapsulation**: Coefficients, covariance matrices stored separately
3. **Clean Interface**: Same `.fit()`, `.predict()`, `.score()`, `.f_test()` methods
4. **No Variable Mix-up**: Each market's state is isolated

## Generated Files
- `market_comparison.png`: Comparative visualizations
- `synthetic_report.md`: Synthetic data verification
- `synthetic_residual_plots.png`: Residual analysis plots
