import numpy as np

def shuffle_data(X, y, seed=None):
	np.random.seed(seed)
	indices = np.random.permutation(len(X))

	X = X[indices]
	y = y[indices]

	return X, y