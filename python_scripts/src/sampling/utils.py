import numpy as np


def compute_prob(data, bin_width):
    bins = np.arange(0, np.max(data), bin_width)
    counts, edges = np.histogram(data, bins=bins, density=True)
    prob = counts * bin_width    
    prob = prob / np.sum(prob)
    return prob
