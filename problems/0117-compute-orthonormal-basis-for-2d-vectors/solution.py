import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    bias = []

    # No vectors → empty basis
    if len(vectors) == 0:
        return bias

    # -------- v1 --------
    v1 = np.array(vectors[0], dtype=float)
    v1_len = np.linalg.norm(v1)

    if v1_len >= tol:
        u1 = v1 / v1_len
        bias.append(u1)
    else:
        # First vector is zero → cannot create basis → stop early
        return bias

    # -------- v2 (only if it exists) --------
    if len(vectors) > 1:
        v2 = np.array(vectors[1], dtype=float)

        # Remove projection onto u1
        v2_proj_u1 = np.dot(v2, u1) * u1
        u2 = v2 - v2_proj_u1

        u2_len = np.linalg.norm(u2)

        if u2_len >= tol:
            u2 = u2 / u2_len
            bias.append(u2)

    return bias
