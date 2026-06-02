
# Week 13 Task B Kaggle Report

## 1. Dataset Information

本任务使用 Week11 的 Kaggle 房价数据 train.csv。

- 样本量：29451
- 字段数：12
- 目标变量：TARGET(PRICE_IN_LACS)
- 每一行代表一套房产记录

这份数据适合练习正则化，因为其中包含面积、房间数、是否在建、是否可入住、经纬度等变量，部分变量之间可能存在相关性。

## 2. Model Performance

| model | RMSE | MAE |
| --- | --- | --- |
| OLS | 587.2208 | 201.7243 |
| Ridge | 587.1699 | 201.2009 |
| Lasso | 587.1888 | 201.4617 |
| ElasticNet | 587.1681 | 201.1803 |

## 3. Best Parameters

| Model | Best Params | Best CV RMSE |
|---|---|---:|
| Ridge | {'model__alpha': np.float64(26.82695795279722)} | 512.2974 |
| Lasso | {'model__alpha': np.float64(0.19306977288832497)} | 512.3012 |
| ElasticNet | {'model__alpha': np.float64(0.0019306977288832496), 'model__l1_ratio': 0.3} | 512.2973 |

## 4. Lasso Selected Features

Lasso 保留下来的变量如下：

| feature | coef |
| --- | --- |
| SQUARE_FT | 470.2139 |
| BHK_NO. | -244.8923 |
| POSTED_BY_Dealer | 199.3253 |
| POSTED_BY_Owner | 192.6659 |
| RESALE | -152.0135 |
| LONGITUDE | -18.4634 |
| READY_TO_MOVE | -14.7655 |
| LATITUDE | -3.7941 |
| RERA | -2.8938 |
| BHK_OR_RK_RK | 1.133 |

## 5. Top 5 Important Factors

如果业务方要求给出最关键的 5 个影响因素，我会参考 Lasso 保留下来的变量以及系数大小。

Top 5 变量为：

['SQUARE_FT', 'BHK_NO.', 'POSTED_BY_Dealer', 'POSTED_BY_Owner', 'RESALE']

## 6. Interpretation

如果正则化方法相比 OLS 没有明显提升，可能原因是：

- 数据中特征数量并不算特别高
- 有些变量本身已经是强预测变量
- 线性模型对房价这种复杂问题表达能力有限
- 房价中存在极端值和地段等隐藏因素

Lasso 剔除变量不一定代表这些变量完全没用，只能说明在当前线性模型和当前数据下，它们对预测贡献较弱。
