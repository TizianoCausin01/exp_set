import torch
import numpy as np

def response_prob(feats_dict):
    p_feat_dict = {}
    for k, v in feats_dict.items():
        n_neu = feats_dict[k][0].shape
        n_stim = len(feats_dict[k])
        stack_resp = np.stack(v, axis=1) 
        freq_fire = (stack_resp > 0).sum(axis=1)
        p_feat_dict[k] = freq_fire.astype(np.float32) / n_stim
    return p_feat_dict

    # end for k, v in feats_dict.items():
