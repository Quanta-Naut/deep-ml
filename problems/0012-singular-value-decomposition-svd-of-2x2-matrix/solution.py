import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    
    eig_val = eigen_val_2x2(A.T@A)

    D = np.diag(eig_val)

    S = np.sqrt(D)

    v1, v2 = np.zeros((2, 1))

    x = eigen_vector_2x2(A.T@A, eig_val[0])
    length = np.sqrt(np.sum(x**2))
    v1 = x / length
    v1 = v1.reshape(-1, 1)

    x = eigen_vector_2x2(A.T@A, eig_val[1])
    length = np.sqrt(np.sum(x**2))
    v2 = x / length
    v2 = v2.reshape(-1, 1)

    V = np.hstack([v1, v2])

    u1 = A@v1 / np.diag(S)[0]
    u2 = A@v2 / np.diag(S)[1]

    U = np.hstack([u1, u2])

    return (U, np.diag(S), V.T)

    pass

def eigen_val_2x2(A):
    a, b = A[0]
    c, d = A[1]

    trace = a + d
    det = a*d - b*c

    eig_val = np.array([
    (trace + np.sqrt(trace**2 - 4*det)) / 2,
    (trace - np.sqrt(trace**2 - 4*det)) / 2
    ]) 

    return eig_val

def eigen_vector_2x2(A, lam):
    M = A - lam*np.eye(2)

    if not np.allclose(M[0], 0):
        a, b = M[0]
    else:
        a, b = M[0]

    if abs(b) > 1e-12:
        x2 = 1
        x1 = -b / a if abs(a) > 1e-12 else 0
    else:
        x1 = 1
        x2 = 0
    
    return np.array([x1, x2])



