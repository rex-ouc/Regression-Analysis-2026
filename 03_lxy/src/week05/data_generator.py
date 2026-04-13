import numpy as np

def generate_design_matrix(n=100, rho=0.0, random_seed=42):
    """
    生成包含两个特征X1和X2的设计矩阵X
    
    参数:
    - n: 样本量
    - rho: X1和X2之间的相关系数（控制共线性程度）
    - random_seed: 随机种子，保证结果可复现
    
    返回:
    - X: n×2的设计矩阵（不包含截距项）
    """
    np.random.seed(random_seed)
    
    # 生成X1 ~ N(0,1)
    X1 = np.random.randn(n)
    
    # 根据相关系数rho生成X2
    # X2 = rho * X1 + sqrt(1 - rho^2) * Z, 其中Z ~ N(0,1)
    Z = np.random.randn(n)
    X2 = rho * X1 + np.sqrt(1 - rho**2) * Z
    
    # 组合成设计矩阵
    X = np.column_stack([X1, X2])
    
    return X

def generate_data_with_fixed_design(X, beta_true, sigma):
    """
    基于固定设计矩阵生成因变量Y
    
    参数:
    - X: 固定的设计矩阵 (n×2)
    - beta_true: 真实参数 [β1, β2]
    - sigma: 噪音的标准差
    
    返回:
    - y: 生成的因变量
    - epsilon: 生成的随机噪音
    """
    n = X.shape[0]
    epsilon = np.random.normal(0, sigma, n)
    y = X @ beta_true + epsilon
    
    return y, epsilon

if __name__ == "__main__":
    # 测试代码
    print("测试数据生成器...")
    
    # 测试正交情况 (rho=0)
    X_orth = generate_design_matrix(n=100, rho=0.0)
    print(f"正交情况 - X1和X2的相关系数: {np.corrcoef(X_orth.T)[0,1]:.4f}")
    
    # 测试共线情况 (rho=0.99)
    X_collinear = generate_design_matrix(n=100, rho=0.99)
    print(f"共线情况 - X1和X2的相关系数: {np.corrcoef(X_collinear.T)[0,1]:.4f}")
    
    # 测试固定设计
    beta_true = np.array([5.0, 3.0])
    sigma = 2.0
    y, epsilon = generate_data_with_fixed_design(X_orth, beta_true, sigma)
    print(f"生成Y的维度: {y.shape}")
    print(f"生成噪音的均值: {epsilon.mean():.4f}, 标准差: {epsilon.std():.4f}")