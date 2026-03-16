import numpy as np


def relative_l2_error(u_pred: np.ndarray, u_true: np.ndarray, eps: float = 1e-12) -> float:
    """
    Relative L2 error: ||u_pred - u_true||_2 / ||u_true||_2
    Works on any shape (flattens internally).
    """
    a = np.ravel(u_pred - u_true)
    b = np.ravel(u_true)
    num = np.linalg.norm(a)
    den = np.linalg.norm(b) + eps
    return float(num / den)