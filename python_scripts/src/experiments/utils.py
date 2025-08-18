import joblib
import numpy as np
import os, sys
from pref_viewing.utils import get_extreme_k, plot_imgs
from parallel.parallel_funcs import print_wise

def project_on(data, basis):
    return data@basis.T
    
def project_onto_PCs(model_name, target_layer, pooling, paths):
    feats_path = f"{paths["results_path"]}/imagenet_val_{model_name}_{target_layer}_{pooling}_features.pkl"
    feats = joblib.load(feats_path)
    PCs_path = f"{paths["results_path"]}/imagenet_val_{model_name}_{target_layer}_{pooling}_pca_model_1000_PCs.pkl"
    PCs = joblib.load(PCs_path)
    dim_redu = project_on(feats, PCs.components_)
    return dim_redu
    
def project_onto_CCs(model_name1, model_name2, target_layer1, target_layer2, pooling, n_components_cca, paths):
    feats_path1 = f"{paths["results_path"]}/imagenet_val_{model_name1}_{target_layer1}_{pooling}_features.pkl"
    feats1 = joblib.load(feats_path1)
    feats_path2 = f"{paths["results_path"]}/imagenet_val_{model_name2}_{target_layer2}_{pooling}_features.pkl"
    feats2 = joblib.load(feats_path2)
    if pooling == "maxpool":
        pca_opt = False
        if feats1.shape[1]>1000:
            feats1 = project_onto_PCs(model_name1, target_layer1, pooling)
            pca_opt = True
        if feats2.shape[1]>1000:
            feats2 = project_onto_PCs(model_name2, target_layer2, pooling)
            pca_opt = True
        if pca_opt:
            cca_path = f"{paths["results_path"]}/cca_{model_name1}_vs_{model_name2}_{pooling}/cca_{model_name1}_vs_{model_name2}_{n_components_cca}_components_pca_{target_layer1}_vs_{target_layer2}.pkl"
        else:
            cca_path = f"{paths["results_path"]}/cca_{model_name1}_vs_{model_name2}_{pooling}/cca_{model_name1}_vs_{model_name2}_{n_components_cca}_components_{target_layer1}_vs_{target_layer2}.pkl"
    else:
        cca_path = f"{paths["results_path"]}/cca_{model_name1}_vs_{model_name2}_{pooling}/cca_{model_name1}_vs_{model_name2}_{n_components_cca}_components_pca_{target_layer1}_vs_{target_layer2}.pkl"
    weights_dict = joblib.load(cca_path)
    d1 = feats1 @ weights_dict["W1"]
    d2 = feats2 @ weights_dict["W2"]
    return d1, d2


def map_on_savenames(model_name, layer_name):
    if model_name == "alexnet":
        model_save_name = "AlexNet"
    elif model_name == "resnet50":
        model_save_name = "ResNet50"
    else:
        print_wise("falling back to the same name")
        model_save_name = model_name
    layer_save_name = layer_name.replace(".", "")
    return model_save_name, layer_save_name


def convert_to_save(imgs_idx, loader):
    img_list = [loader[i][0] for i in imgs_idx]
    return img_list


def get_k_imgs(data, loader, k, dim, extreme, show_opt=False):
    idx = get_extreme_k(data, loader, k, dim, extreme)
    img_list = convert_to_save(idx, loader)
    if show_opt:
        img_list_show = [img.permute(1,2,0) for img in img_list]
        _ = plot_imgs(img_list_show)
    return idx, img_list


