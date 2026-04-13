import numpy as np

#1.AnalyticalSolver:使用正规方程求解
class AnalyticalSolver:
 def __init__(self):
  pass

 def solve(self,X,y):
  """
  使用正规方程(Normal Equation)求解线性回归问题。
  X:输入特征矩阵(m x n)
  y:目标值(m x 1)
  """
  #使用 np.linalg.solve 求解正规方程 X.T*X*beta=X.T*y
  #beta是线性回归的参数
  beta=np.linalg.solve(X.T.dot(X),X.T.dot(y))
  return beta

#2.GradientDescentSolver:使用批量梯度下降法求解
class GradientDescentSolver:
 def __init__(self,learning_rate=0.01,epochs=1000):
  """
  初始化梯度下降求解器。
  learning_rate:学习率
  epochs:迭代次数
  """
  self.learning_rate=learning_rate
  self.epochs=epochs

 def solve(self,X,y):
  """
  使用批量梯度下降法(Batch Gradient Descent)求解线性回归问题。
  X:输入特征矩阵(m x n)
  y:目标值(m x 1)
  """
  #初始化参数beta为零
  m,n=X.shape
  beta=np.zeros(n)

  #迭代进行梯度下降
  for _ in range(self.epochs):
   #计算预测值
   predictions=X.dot(beta)
   #计算误差
   errors=predictions-y
   #计算梯度
   gradient=(1/m)*X.T.dot(errors)
   #更新beta
   beta-=self.learning_rate*gradient

  return beta