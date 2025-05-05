import os
import numpy as np


def load_valid_input(k_str, file_name):
    if not float(k_str).is_integer():
        raise ValueError
    k = int(float(k_str))
    if file_name[-4:] != ".txt" or os.path.getsize(file_name) == 0:
        raise ValueError
    x_matrix = np.genfromtxt(file_name, delimiter=',')
    # File must contain only valid float values, and k<n
    if x_matrix.size == 0 or np.isnan(x_matrix).any() or k >= x_matrix.shape[0]:
        raise ValueError
    return k, x_matrix.tolist()

def init_h_matrix(n, k, w_matrix):
    m = np.mean(w_matrix)
    return np.random.uniform(0, 2 * np.sqrt(m / k), size=(n, k)).tolist()
