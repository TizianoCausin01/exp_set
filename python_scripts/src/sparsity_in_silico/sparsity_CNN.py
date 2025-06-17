import torch


def response_prob(feats_dict):
    p_feat_dict = {}
    for k, v in feats_dict.items():
        print(k)
        print(v[0].size())
        n_neu = feats_dict[k][0].size()
        n_stim = len(feats_dict[k])
        freq_fire = torch.zeros(n_neu, dtype=torch.uint8)
        for risp in v:
            bool_risp = (risp > 0).to(dtype=torch.uint8)
            freq_fire += bool_risp.to(dtype=torch.uint8)
        # end for risp in v:
        p_feat_dict[k] = freq_fire / n_stim
    return p_feat_dict

    # end for k, v in feats_dict.items():
