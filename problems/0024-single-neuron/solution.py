import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):

	features = np.array(features)
	labels = np.array(labels)
	weights = np.array(weights)
	bias = bias

	weighted_sum = []

	for feature in features:
		weighted_sum.append(sigmoid(feature@weights + bias))

	weighted_sum = np.array(weighted_sum)

	mse_vals = [] 

	for i, label in zip(weighted_sum, labels):
		mse_vals.append((label - i)**2)

	mse_vals = np.array(mse_vals)
	mse = np.sum(mse_vals) / len(mse_vals)

	probabilities = weighted_sum
	
	return probabilities, mse

def sigmoid(z):
	return 1 / (1 + math.exp(-z))








