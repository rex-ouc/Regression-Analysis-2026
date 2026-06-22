import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
# ====================== A1 作业要求：生成低秩隐因子高维数据 ======================
def generate_highdim_latent_data(
    n_samples: int,
    n_features: int,
    n_latent: int,
    noise_std: float = 0.3,
    random_seed: int = 42
):
    """
    生成带隐因子的高维冗余数据（满足A1：p>n、低秩隐因子结构）
    :param n_samples: 样本量 ≥120
    :param n_features: 特征数 ≥60
    :param n_latent: 隐因子数量（远小于p）
    :return: X, y, latent_factors
    """
    rng = np.random.default_rng(random_seed)
    # 1.生成少量隐因子 latent factors
    latent = rng.normal(size=(n_samples, n_latent))
    # 2.载荷矩阵：隐因子线性组合生成全部原始特征（制造多重共线性）
    loadings = rng.normal(size=(n_latent, n_features))
    X = latent @ loadings + noise_std * rng.normal(size=(n_samples, n_features))
    # 3.y仅由隐因子线性生成，不由原始变量独立驱动
    beta_latent = rng.normal(size=n_latent)
    y = latent @ beta_latent + noise_std * rng.normal(size=n_samples)
    return X, y, latent

# ====================== C1场景1：稀疏真值数据生成 ======================
def generate_sparse_truth_data(
    n_samples: int,
    n_features: int,
    n_signal_feats: int,
    noise_std: float = 0.3,
    random_seed: int = 42
):
    """稀疏场景：仅少数原始变量有效，其余纯噪声"""
    rng = np.random.default_rng(random_seed)
    X = rng.normal(size=(n_samples, n_features))
    beta = np.zeros(n_features)
    signal_idx = rng.choice(np.arange(n_features), size=n_signal_feats, replace=False)
    beta[signal_idx] = rng.normal(size=n_signal_feats)
    y = X @ beta + noise_std * rng.normal(size=n_samples)
    return X, y, signal_idx

# ====================== 新增：二分类模拟数据生成 Week15 ======================
def generate_synthetic_binary_taskA(n_samples=500, n_feats=5, seed=42):
    """Task A：标准逻辑回归DGP数据，≥4特征，至少2个有效特征，伯努利采样y"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_feats))
    # 设定真实系数：前两特征显著
    beta = np.array([1.8, -1.5, 0.4, 0.0, 0.0])
    eta = X @ beta
    # Sigmoid映射概率
    p = 1 / (1 + np.exp(-np.clip(eta, -100, 100)))
    # 伯努利抽样0/1标签
    y = rng.binomial(n=1, p=p, size=n_samples)
    import pandas as pd
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_feats)])
    df["y"] = y
    df["true_prob"] = p
    return df, beta


def generate_highdim_collinear_binary_taskD(n_samples=600, n_feats=22, seed=42):
    """Task D：高维+共线性+噪声二分类数据，特征≥20"""
    rng = np.random.default_rng(seed)
    base_latent = rng.normal(size=(n_samples, 3))
    X = np.zeros((n_samples, n_feats))
    X[:, :3] = base_latent
    # 构造强共线性特征
    X[:, 3] = base_latent[:,0] * 0.8 + rng.normal(n_samples)*0.05
    X[:, 4] = base_latent[:,1] * 1.2 + rng.normal(n_samples)*0.05
    # 剩余纯噪声特征
    X[:, 5:] = rng.normal(size=(n_samples, n_feats - 5))
    # 真实稀疏系数
    beta = np.zeros(n_feats)
    beta[:3] = [2.0, -1.6, 0.9]
    eta = X @ beta
    p = 1 / (1 + np.exp(-np.clip(eta, -100, 100)))
    y = rng.binomial(1, p, n_samples)
    import pandas as pd
    cols = [f"f{i}" for i in range(n_feats)]
    df = pd.DataFrame(X, columns=cols)
    df["y"] = y
    return df
