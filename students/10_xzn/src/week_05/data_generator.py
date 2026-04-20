import numpy as np

def generate_design_matrix(n_samples: int, rho: float) -> np.ndarray:

    z1 = np.random.normal(loc=0, scale=1, size=n_samples)
    z2 = np.random.normal(loc=0, scale=1, size=n_samples)
    
    x1 = z1
    x2 = rho * x1 + np.sqrt(1 - rho**2) * z2
    
    X = np.column_stack((x1, x2))
    return X