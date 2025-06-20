import torch
import numpy as np

def response_prob_np(feats_dict):
    p_feat_dict = {}
    for k, v in feats_dict.items():
        n_neu = feats_dict[k][0].shape
        n_stim = len(feats_dict[k])
        stack_resp = np.stack(v, axis=1) 
        freq_fire = (stack_resp > 0).sum(axis=1)
        p_feat_dict[k] = freq_fire.astype(np.float32) / n_stim
    return p_feat_dict

    # end for k, v in feats_dict.items():

def response_prob_torch(feats_dict):
    p_feat_dict = {}
    for k, v in feats_dict.items():
        n_neu = feats_dict[k][0].size()
        n_stim = len(feats_dict[k])
        freq_fire = torch.zeros(n_neu, dtype=torch.uint8)
        stack_resp = torch.stack(v, dim=1) 
        freq_fire = (stack_resp > 0).sum(dim=1)
        p_feat_dict[k] = freq_fire.float() / n_stim
    return p_feat_dict
