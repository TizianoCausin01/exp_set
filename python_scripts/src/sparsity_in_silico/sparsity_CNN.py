import numpy as np
from .utils import participation_ratio


"""
response_prob
Computes the response probability of an ANN given its activations for a set of stimuli. 
INPUT:
- data_mat: numpy.ndarray -> data matrix with responses of neurons (neu x stim)

OUTPUT:
- p_feat: numpy.ndarray -> each neuron's probability to fire given the dataset (neu x 1)
"""
def response_prob(data_mat):
    n_neu, n_stim = data_mat.shape
    freq_fire = (data_mat > 0).sum(axis=1)
    p_feat = freq_fire.astype(np.float32) / n_stim
    return p_feat
# EOF


"""
rust_sparseness
Computes the sparseness as in Rust & DiCarlo 2012
S = (1 - a) / (1 - 1/N)     [a = participation ratio ; N = num stim]
INPUT:
- data_mat: numpy.ndarray -> data matrix with responses of neurons (neu x stim)

OUTPUT:
- S: numpy.ndarray -> each neuron's sparseness (neu x 1)

"""
def rust_sparseness(data_mat):
    n_neu, n_stim = data_mat.shape
    a = participation_ratio(data_mat)
    S = (1 - a) / (1 - 1/n_stim)
    return S
# EOF
