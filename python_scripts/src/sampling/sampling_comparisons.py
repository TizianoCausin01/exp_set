import numpy as np

"""
kld_calc
p1 = reference distribution
p2 = approximating distribution
kld -> how much info is lost when using q to approximate p
"""

def kld_calc(p, q, eps=10e-12):
    p_list = [p, q]
    idx_larger = np.argmax([p_list[0].shape[0], p_list[1].shape[0]]) # todo check if they are the same
    idx_smaller = 1 - idx_larger
    diff_size = p_list[idx_larger].shape[0] - p_list[idx_smaller].shape[0]
    to_fill_in = np.zeros(diff_size)
    p_list[idx_smaller] = np.concatenate([p_list[idx_smaller], to_fill_in])
    idx_to_remove = np.where(p_list[0] == 0) # removes the zero entries in p because kld -> 0 for p->0
    p_list[0] = np.delete(p_list[0], idx_to_remove)
    p_list[1] = np.delete(p_list[1], idx_to_remove)
    p_list[1] = p_list[1] + eps
    kld = p_list[0] @ np.log2(p_list[0]/ p_list[1])
    return kld


