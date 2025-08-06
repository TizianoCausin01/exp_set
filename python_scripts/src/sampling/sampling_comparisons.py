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


"""
sum_for_var
computes the two sums to be added to compute the variance
batch = data (n x D)

curr_sum, curr_sum_sq -> the sum and the sum of squares respectively
"""
def sum_for_var(batch):
    curr_sum = np.sum(batch, axis=0)
    curr_sum_sq = np.sum(batch**2, axis=0)
    return curr_sum, curr_sum_sq

def variance(tot_sum, tot_sum_sq, n):
    E_X = tot_sum / n
    E_X2 = tot_sum_sq / n
    var_per_dim = (E_X2 - E_X**2) * n / (n - 1) # n/(n-1) correction 
    #var_per_dim = tot_sum_sq/(n) - (tot_sum /(n))**2 # var(X) = E[X^2] - E[X]^2 , n-1 for the number of stimuli, np.mean along the features
    # in case you wanted an overall summary statistics, it'd just np.mean(var_per_dim), the average of the variance across the dimensions
    return var_per_dim 
