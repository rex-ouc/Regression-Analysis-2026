"""
Week4 Assignment:TheTaleofTwoSolvers
Author:zhouying
"""
from solvers import AnalyticalSolver,GradientDescentSolver
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,SGDRegressor
import statsmodels.api as sm

#任务二
#设置随机种子
np.random.seed(42)
#生成低维数据（实验A）
N_low=10000
P_low=10
X_low=np.random.randn(N_low,P_low)
y_low=np.dot(X_low,np.random.randn(P_low))+np.random.randn(N_low)

#生成高维数据（实验B，P=2000）
N_high=10000
P_high=2000
X_high=np.random.randn(N_high,P_high)
y_high=np.dot(X_high,np.random.randn(P_high))+np.random.randn(N_high)

#定义计算时间和MSE的函数
def compute_performance(solver,X,y):
 start_time=time.time()
 beta=solver.solve(X,y)
 execution_time=time.time()-start_time
 predictions=X.dot(beta)
 mse=mean_squared_error(y,predictions)
 return execution_time,mse

#使用AnalyticalSolver和GradientDescentSolver分别计算低维和高维数据
analytical_solver=AnalyticalSolver()
gd_solver=GradientDescentSolver(learning_rate=0.01,epochs=1000)

#对低维数据计算
time_low_analytical,mse_low_analytical=compute_performance(analytical_solver,X_low,y_low)
time_low_gd,mse_low_gd=compute_performance(gd_solver,X_low,y_low)

#对高维数据计算
time_high_analytical,mse_high_analytical=compute_performance(analytical_solver,X_high,y_high)
time_high_gd,mse_high_gd=compute_performance(gd_solver,X_high,y_high)

#输出任务二结果
print(f"Low-dimensional(P=10)-AnalyticalSolverTime:{time_low_analytical:.4f}s,MSE:{mse_low_analytical:.4f}")
print(f"Low-dimensional(P=10)-GradientDescentSolverTime:{time_low_gd:.4f}s,MSE:{mse_low_gd:.4f}")
print(f"High-dimensional(P=2000)-AnalyticalSolverTime:{time_high_analytical:.4f}s,MSE:{mse_high_analytical:.4f}")
print(f"High-dimensional(P=2000)-GradientDescentSolverTime:{time_high_gd:.4f}s,MSE:{mse_high_gd:.4f}")

#任务三：工业界API对比实验
def compute_time_and_mse(model_func,X,y,model_type="sklearn"):
 start_time=time.time()
 if model_type=="statsmodels":
  X_with_const=sm.add_constant(X)
  model=model_func(X,y)
  predictions=model.predict(X_with_const)
 else:
  model=model_func(X,y)
  predictions=model.predict(X)
 execution_time=time.time()-start_time
 mse=mean_squared_error(y,predictions)
 return execution_time,mse

#1.StatsmodelsOLS函数（适配修复后的compute_time_and_mse）
def statsmodels_ols(X,y):
 X_with_const=sm.add_constant(X)
 model=sm.OLS(y,X_with_const)
 return model.fit()

#2.sklearnLinearRegression函数
def sklearn_lr(X,y):
 model=LinearRegression()
 model.fit(X,y)
 return model

#3.sklearnSGDRegressor函数
def sklearn_sgd(X,y):
 model=SGDRegressor(max_iter=1000,tol=1e-3,random_state=42)
 model.fit(X,y)
 return model

#对高维数据进行回归并记录时间和MSE
time_ols,mse_ols=compute_time_and_mse(statsmodels_ols,X_high,y_high,model_type="statsmodels")
time_lr,mse_lr=compute_time_and_mse(sklearn_lr,X_high,y_high)
time_sgd,mse_sgd=compute_time_and_mse(sklearn_sgd,X_high,y_high)

#输出任务三结果
print(f"\nStatsmodelsOLSTime:{time_ols:.4f}s,MSE:{mse_ols:.4f}")
print(f"SklearnLinearRegressionTime:{time_lr:.4f}s,MSE:{mse_lr:.4f}")
print(f"SklearnSGDRegressorTime:{time_sgd:.4f}s,MSE:{mse_sgd:.4f}")

print("\n✅第4周作业全部运行完成！")