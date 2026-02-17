def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	
	import numpy as np

	arr = np.array(matrix)

	vals = np.linalg.eigvals(arr)

	return list(vals)