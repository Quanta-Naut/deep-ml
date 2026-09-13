import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix.
    
    Args:
        A: 2x2 numpy array
    
    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors)
    """
    eig_vals = eig_val(A.T@A)

    v1 = eig_vec(A.T@A, eig_vals[0])
    v2 = eig_vec(A.T@A, eig_vals[1])

    u1 = A@v1 / np.sqrt(eig_vals[0])
    u2 = A@v2 / np.sqrt(eig_vals[1])

    s = np.sqrt(eig_vals)
    V = np.column_stack((v1, v2))
    U = np.column_stack((u1, u2))

    return (U, s, V.T)

def eig_val(A):
    a, b = A[0]
    c, d = A[1]

    eig_vals = np.zeros((2))

    eig_vals[0] = ((a + d) + np.sqrt((a + d)**2 - 4*(a*d - b*c))) / 2
    eig_vals[1] = ((a + d) - np.sqrt((a + d)**2 - 4*(a*d - b*c))) / 2

    return eig_vals

def eig_vec(A, lam):
    a, b = A[0]
    c, d = A[1]

    v = np.array([-b, a - lam])

    if np.sum(v**2) == 0:
        v = np.array([d - lam, -c])

    norm = np.sqrt(np.sum(v**2))
    return v / norm


