import numpy as np

# Example 1D array (vector)
observation_1d = np.array([1, 2, 3, 4, 5])

# Adding a new axis to make it a 2D row vector
observation_2d = observation_1d[np.newaxis, :]

# Example 2D array (matrix)
observation_2d_matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Adding a new axis to make it a 3D array
observation_3d = observation_2d_matrix[np.newaxis, :]

observation_1d.shape, observation_2d.shape, observation_2d_matrix.shape, observation_3d.shape
print(observation_1d, observation_2d, observation_2d_matrix, observation_3d)
