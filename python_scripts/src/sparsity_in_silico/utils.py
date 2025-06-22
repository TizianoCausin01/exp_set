"""utils module"""
import numpy as np






def get_data_mat(data_dict, key, dtype=np.float32):
    data_mat = data_dict[key][:].astype(dtype)
    data_mat = np.transpose(data_mat)
    return data_mat

"""
participation_ratio
Computes the participation ratio of each neuron in data_mat (rows). 
participation_ratio = avg_rate^2 / avg_squared_rate
INPUT:
- data_mat: numpy.ndarray -> data matrix with responses of neurons (neu x stim)

OUTPUT:
- a: numpy.ndarray -> each neuron's participation ratio (neu x 1)
"""
def participation_ratio(data_mat: np.ndarray) -> np.ndarray:
    n_neu, n_stim = data_mat.shape
    numerator = (np.sum(data_mat, axis=1)/n_stim)**2
    eps = 10e-8 # to avoid dividing by zero
    denominator = np.sum(data_mat**2 + eps, axis=1)/n_stim
    a = numerator/denominator
    return a
# EOF
