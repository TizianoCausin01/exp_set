from datetime import datetime
import numpy as np
import torch
from dim_redu_anns.utils import get_relevant_output_layers, get_layer_out_shape
from parallel.parallel_funcs import print_wise
def compute_prob(data, bin_width):
    bins = np.arange(0, np.max(data), bin_width)
    counts, edges = np.histogram(data, bins=bins, density=True)
    prob = counts * bin_width    
    prob = prob / np.sum(prob)
    return prob


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
#EOF

def variance(tot_sum, tot_sum_sq, n):
    E_X = tot_sum / n
    E_X2 = tot_sum_sq / n
    var_per_dim = (E_X2 - E_X**2) * n / (n - 1) # n/(n-1) correction 
    #var_per_dim = tot_sum_sq/(n) - (tot_sum /(n))**2 # var(X) = E[X^2] - E[X]^2 , n-1 for the number of stimuli, np.mean along the features
    # in case you wanted an overall summary statistics, it'd just np.mean(var_per_dim), the average of the variance across the dimensions
    return var_per_dim 
#EOF

def variance_estimation_loop(feature_extractor, target_layer, loader, num_stim, batch_size):
    if num_stim == 0:
        num_stim = 50000
    # if num_stim == 0:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = get_layer_out_shape(feature_extractor, target_layer)
    tot_s = np.zeros(np.prod(d))
    tot_ss = np.zeros(np.prod(d))
    counter = 0
    for inputs, _ in loader:
        counter += 1
        if counter*batch_size > num_stim:
            break
        # if counter*batch_size > num_stim:
        print(datetime.now().strftime("%H:%M:%S"), f"starting batch {counter}", flush=True)
        with torch.no_grad():
            inputs = inputs.to(device)
            feats = feature_extractor(inputs)[target_layer]
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            s, ss = sum_for_var(feats)
            tot_s += s
            tot_ss += ss
        # with torch.no_grad():
    # for inputs, _ in loader:
    var = variance(tot_s, tot_ss, num_stim)
    print("average variance", np.mean(var))
    return var
#EOF






def features_sampling(loader, test_feature_extractor, test_layer, random_neu_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counter = 0
    all_feats = []
    for inputs, _ in loader:
        counter += 1
#        print_wise(f"starting batch {counter}")
        with torch.no_grad():
            inputs = inputs.to(device)
            feats = test_feature_extractor(inputs)[test_layer]
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            feats = feats[:, random_neu_idx]
            all_feats.append(feats)
    # end for inputs, _ in loader:
    all_acts = np.concatenate(all_feats, axis=0)
    return all_acts
