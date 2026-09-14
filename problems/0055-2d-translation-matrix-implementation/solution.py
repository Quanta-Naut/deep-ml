import numpy as np
def translate_object(points, tx, ty):

	translated_points = []

	for i in points:
		translated_points.append([i[0] + tx, i[1] + ty])

	return translated_points
