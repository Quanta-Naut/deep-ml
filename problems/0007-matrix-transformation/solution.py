import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	
	a_arr = np.array(A)
	t_arr = np.array(T)
	s_arr = np.array(S)

	if t_arr.shape[0] != t_arr.shape[1] or s_arr.shape[0] != s_arr.shape[1]:
		return -1
	
	if np.linalg.matrix_rank(t_arr) != t_arr.shape[0]:
		return -1

	if np.linalg.matrix_rank(s_arr) != s_arr.shape[0]:
		return -1
	
	t_inv = np.linalg.inv(t_arr)

	return (t_inv @ a_arr @ s_arr).tolist()