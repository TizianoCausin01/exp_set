import os, sys, time
import numpy as np
import joblib
import torch
from sklearn.cross_decomposition import CCA
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from datetime import datetime
sys.path.append("..")
from dim_redu_anns.utils import get_relevant_output_layers



def CCA_loop_within_mod(model_name, pooling, num_components, res_path):
    layer_names = get_relevant_output_layers(model_name)
    cca_dir = f"{res_path}/cca_{model_name}_{pooling}"
    os.makedirs(cca_dir, exist_ok=True)
    layers_RSA = np.zeros((len(layer_names),len(layer_names)))
    for layer_idx1 in range(len(layer_names)):
        for layer_idx2 in range(layer_idx1):
            print(datetime.now().strftime("%H:%M:%S"), f"stating layers {target_layer1} vs {target_layer2}")
            target_layer1 = layer_names[layer_idx1]
            feats_path1 = f"{res_path}/imagenet_val_{model_name}_{target_layer1}_{pooling}_features.pkl"
            target_layer2 = layer_names[layer_idx2]
            feats_path2 = f"{res_path}/imagenet_val_{model_name}_{target_layer2}_{pooling}_features.pkl"
            save_path = f"{cca_dir}/cca_{model_name}_{num_components}_components_{target_layer1}_vs_{target_layer2}.pkl"
            if os.path.exists(save_path):
                print(
                    datetime.now().strftime("%H:%M:%S"),
                    f"CCA already exists for {target_layer1} vs {target_layer2}  at {save_path}",
                    flush=True
                )
            else:
                all_acts1 = joblib.load(feats_path1)
                all_acts2 = joblib.load(feats_path2)
                cca = CCA(n_components = num_components)
                cca.fit(all_acts1, all_acts2)
                weights_dict = {}
                weights_dict["W1"] = cca.x_weights_  # shape: (n_features1, n_components)
                weights_dict["W2"] = cca.y_weights_  # shape: (n_features2, n_components)
                
                # 3. Project the data manually (optional, equivalent to fit_transform)
                d1 = all_acts1 @ weights_dict["W1"]
                d2 = all_acts2 @ weights_dict["W2"]
                coefs_CCA = np.array([
                    np.corrcoef(d1[:, i], d2[:, i])[0, 1] for i in range(d1.shape[1])
                ])
                weights_dict["coefs"] = coefs_CCA
                joblib.dump(weights_dict, save_path)
                layers_RSA[layer_idx1, layer_idx2] = np.mean(coefs_CCA)
                print(datetime.now().strftime("%H:%M:%S"), f"{target_layer1} vs {target_layer2} corr {np.round(np.mean(coefs_CCA), 3)}", flush=True)
    csv_save_path = f"{cca_dir}/{model_name}_similarity_layers.csv"
    np.savetxt(csv_save_path, layers_RSA, delimiter=",")

