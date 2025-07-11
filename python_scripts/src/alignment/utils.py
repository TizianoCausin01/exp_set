import numpy as np 

def get_maxpool_evecs(data, layer_name, layer_shape):
    all_PCs_shape = (data.n_components,) + layer_shape
    evecs = data.components_
    unflat_evecs = np.reshape(evecs, all_PCs_shape)
    if layer_name == 'avgpool' or 'classifier' in layer_name:
        unflat_evecs = np.squeeze(unflat_evecs)
        return unflat_evecs # don't do anything, it's already flat
    else:
        max_evecs = np.max(unflat_evecs, axis=(2,3)) # pools the max in the feats
        return max_evecs
