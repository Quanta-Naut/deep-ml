import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	min_max_cols = []
	std_cols = []

	for col in np.hsplit(data, data.shape[1]):
		min_max_cols.append(fsi_min_max(col))
		std_cols.append(fsi_std(col))

	standardized_data = np.hstack(std_cols)
	normalized_data = np.hstack(min_max_cols)

	return standardized_data, normalized_data

def mean(A):
	return np.sum(A) / len(A)

def std_dev(A):
	mean_val = mean(A)

	return np.sqrt(np.sum((A - mean_val)**2) / len(A))

def fsi_std(A):
	std_vals = np.zeros(A.shape)

	mean_val = mean(A)
	std_dev_val = std_dev(A)

	for i in range(len(A)):
		std_vals[i] = (A[i] - mean_val) / std_dev_val
	
	return std_vals

def fsi_min_max(A):
	min_max_vals = np.zeros(A.shape)

	min_val = np.min(A)
	max_val = np.max(A)

	for i in range(len(A)):
		min_max_vals[i] = (A[i] - min_val)/(max_val - min_val)
	
	return min_max_vals




