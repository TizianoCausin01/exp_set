import torch
import numpy as np
from .utils import participation_ratio


"""
response_prob_np
Computes the response probability of an ANN given its activations for a set of stimuli. With numpy
INPUT: 
- feats_dict: dict{str, np.ndarray} -> the dict with the features at each given layer

OUTPUT
- p_feat_dict: dict{str, np.ndarray} -> the dict with the probability of firing at each given layer
"""
def response_prob_np(feats_dict):
    p_feat_dict = {}
    for k, v in feats_dict.items():
        n_neu = feats_dict[k][0].shape
        n_stim = len(feats_dict[k])
        stack_resp = np.stack(v, axis=1) 
        freq_fire = (stack_resp > 0).sum(axis=1)
        p_feat_dict[k] = freq_fire.astype(np.float32) / n_stim
    # end for k, v in feats_dict.items():
    return p_feat_dict
# EOF

def response_prob_torch(feats_dict):
    p_feat_dict = {}
    for k, v in feats_dict.items():
        n_neu = feats_dict[k][0].size()
        n_stim = len(feats_dict[k])
        freq_fire = torch.zeros(n_neu, dtype=torch.uint8)
        stack_resp = torch.stack(v, dim=1) 
        freq_fire = (stack_resp > 0).sum(dim=1)
        p_feat_dict[k] = freq_fire.float() / n_stim
    # end for k, v in feats_dict.items():
    return p_feat_dict
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
