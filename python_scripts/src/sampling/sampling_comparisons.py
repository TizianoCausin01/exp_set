import os, yaml, sys
from sklearn.cluster import KMeans
import torch
from torchvision import models,datasets, transforms
import numpy as np
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from dadapy import data
from sklearn.decomposition import PCA
import joblib
from dim_redu_anns.utils import get_layer_out_shape
from parallel.parallel_funcs import print_wise
from sampling.sampling_methods import multistage_kmeans, assign_clusters_in_batches, subset_loader, sample_cluster_wise
from sampling.utils import features_sampling
from torch.utils.data import DataLoader, Subset
from alignment.utils import get_usual_transform, get_transform_to_show

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


def ID_var_estimate(model_name, layer_name, feats_projection, n_clusters_per_level, test_network, test_layer, neurons_perc, batch_size, paths, alignment=False, model_name2=None, layer_name2=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model_cls = getattr(models, test_network)
    test_model = test_model_cls(pretrained=True).to(device).eval()
    test_feature_extractor = create_feature_extractor(test_model, return_nodes=[test_layer]).to(
        device
    )
    tot_neurons = np.prod(get_layer_out_shape(test_feature_extractor, test_layer))
    rand_neu_num = int(tot_neurons*neurons_perc/100)
    random_neu_idx = np.random.choice(tot_neurons, rand_neu_num, replace=False)

    centers, labels = multistage_kmeans(feats_projection, n_clusters_per_level, init='k-means++', max_iter=1000, tol=1e-5, n_init=10)
    C = centers[-1]

    final_labels = assign_clusters_in_batches(feats_projection, C, batch_size=1000)

    representatives = sample_cluster_wise(final_labels, 1)
    print_wise(representatives)
    img_num = len(representatives)
    random_idx = np.random.choice(49152, img_num, replace=False)
    print_wise(random_idx)
    imagenet_val_path = f"{paths['data_path']}/imagenet/val"
    loader_kmeans = subset_loader(imagenet_val_path, representatives, batch_size, shuffle=True, to_show=False)
    loader_random = subset_loader(imagenet_val_path, random_idx, batch_size, shuffle=True, to_show=False)
    print_wise("extracting features...")
    all_acts_kmeans = features_sampling(loader_kmeans, test_feature_extractor, test_layer, random_neu_idx)
    all_acts_rand = features_sampling(loader_random, test_feature_extractor, test_layer, random_neu_idx)
                                      
    _data_kmeans = data.Data(all_acts_kmeans)
    id_twoNN_kmeans, _, r = _data_kmeans.compute_id_2NN()


    kmeans_pca = PCA(n_components=min(rand_neu_num, img_num))
    kmeans_pca.fit(all_acts_kmeans)
    kmeans_var = np.var(all_acts_kmeans)
    kmeans_dyn_range = np.mean(np.max(all_acts_kmeans, axis=0))
    _data_rand = data.Data(all_acts_rand)
    id_twoNN_rand, _, r = _data_rand.compute_id_2NN()
    rand_pca = PCA(n_components=min(rand_neu_num, img_num))
    rand_pca.fit(all_acts_rand)
    rand_var = np.var(all_acts_rand)
    rand_dyn_range = np.mean(np.max(all_acts_rand, axis=0))
    print("variance : \n", "kmeans : ", kmeans_var, "\n random : ",  rand_var)
    print("ID : \n", "kmeans : ", id_twoNN_kmeans, "\n random : ",  id_twoNN_rand)
    print("dynamic range : \n", "kmeans : ", kmeans_dyn_range, "\n random : ",  rand_dyn_range)
    to_save_kmeans = {'variance' : kmeans_var, 'ID' : id_twoNN_kmeans, 'PCA' : kmeans_pca, 'sample_imgs' : representatives, 'sample_neurons' : random_neu_idx}
    to_save_random = {'variance' : rand_var, 'ID' : id_twoNN_rand, 'PCA' : rand_pca, 'sample_imgs' : random_idx, 'sample_neurons' : random_neu_idx}
    if alignment == True:
        kmeans_path = f"{paths['results_path']}/sampling_comparisons/kmeans_CCs_{model_name}+{model_name2}_{layer_name}+{layer_name2}_test_{test_network}_{test_layer}_{n_clusters_per_level[-1]}_samples_{neurons_perc}perc_neurons.pkl" 
        random_path = f"{paths['results_path']}/sampling_comparisons/random_PCs_{model_name}+{model_name2}_{layer_name}+{layer_name2}_test_{test_network}_{test_layer}_{n_clusters_per_level[-1]}_samples_{neurons_perc}perc_neurons.pkl" 
    else:
        kmeans_path = f"{paths['results_path']}/sampling_comparisons/kmeans_PCs_{model_name}_{layer_name}_test_{test_network}_{test_layer}_{n_clusters_per_level[-1]}_samples_{neurons_perc}perc_neurons.pkl" 
        random_path = f"{paths['results_path']}/sampling_comparisons/random_PCs_{model_name}_{layer_name}_test_{test_network}_{test_layer}_{n_clusters_per_level[-1]}_samples_{neurons_perc}perc_neurons.pkl"
    joblib.dump(to_save_kmeans, kmeans_path)
    joblib.dump(to_save_random, random_path)
