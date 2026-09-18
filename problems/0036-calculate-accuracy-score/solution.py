import numpy as np

def accuracy_score(y_true, y_pred):
	score = 0
	length = len(y_true)

	for i, j in zip(y_true, y_pred):
		if i == j:
			score += 1

	return score / length