import numpy as np

def generate_design_matrix(n_samples: int = 1000, rho: float = 0.0, seed: int = 42) -> np.ndarray:
    """
    生成带指定相关系数rho的特征矩阵X(含常数项)
    
    参数:
        n_samples: 样本数量
        rho: X1和X2的相关系数,范围[-1,1]
        seed: 随机种子,保证可复现
    返回:
        X: n_samples x 3 的设计矩阵,第一列为常数项
    """
    np.random.seed(seed)
    
    # 生成正交的标准正态变量
    z1 = np.random.randn(n_samples)
    z2 = np.random.randn(n_samples)
    
    # 构造相关变量X1, X2（均值为0,方差为1）
    x1 = z1
    x2 = rho * z1 + np.sqrt(1 - rho**2) * z2
    
    # 添加常数项,组成设计矩阵
    X = np.column_stack([np.ones(n_samples), x1, x2])
    return X

if __name__ == "__main__":
    # 测试函数,打印相关系数验证
    X_test = generate_design_matrix(rho=0.99)
    print("X1和X2的相关系数:", np.corrcoef(X_test[:,1], X_test[:,2])[0,1])