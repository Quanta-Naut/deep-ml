import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.zeros_like(b)

    for _ in range(n):
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])

            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        
        x = x_new
    
    return [round(val, 4) for val in x.tolist()]