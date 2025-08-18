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


def CCA_loop_within_mod(model_name, pooling, num_components, pca_option, res_path):
    layer_names = get_relevant_output_layers(model_name)
    cca_dir = f"{res_path}/cca_{model_name}_{pooling}"
    os.makedirs(cca_dir, exist_ok=True)
    layers_RSA = np.zeros((len(layer_names), len(layer_names)))
    for layer_idx1 in range(len(layer_names)):
        target_layer1 = layer_names[layer_idx1]
        feats_path1 = f"{res_path}/imagenet_val_{model_name}_{target_layer1}_{pooling}_features.pkl"
        print(
            datetime.now().strftime("%H:%M:%S"),
            f"starting loading {target_layer1}",
            flush=True,
        )
        all_acts1 = joblib.load(feats_path1)
        print(
            datetime.now().strftime("%H:%M:%S"),
            f"finished loading {target_layer1}",
            flush=True,
        )
        if pca_option == True:
            first_projection = False
            if all_acts1.shape[1] > 1000:
                PCs_path1 = f"{res_path}/imagenet_val_{model_name}_{target_layer1}_{pooling}_pca_model_1000_PCs.pkl"
                print(
                    datetime.now().strftime("%H:%M:%S"),
                    f"starting loading PC1",
                    flush=True,
                )
                PCs1 = joblib.load(PCs_path1)
                print(
                    datetime.now().strftime("%H:%M:%S"),
                    f"starting loading PC1",
                    flush=True,
                )
                all_acts1 = all_acts1 @ PCs1.components_.T
                print(
                    datetime.now().strftime("%H:%M:%S"),
                    f"finished backprojecting in PCs1",
                    flush=True,
                )
                first_projection = True

        for layer_idx2 in range(layer_idx1):
            target_layer2 = layer_names[layer_idx2]
            print(
                datetime.now().strftime("%H:%M:%S"),
                f"starting layers {target_layer1} vs {target_layer2}",
            )
            feats_path2 = f"{res_path}/imagenet_val_{model_name}_{target_layer2}_{pooling}_features.pkl"
            all_acts2 = joblib.load(feats_path2)
            save_path = f"{cca_dir}/cca_{model_name}_{num_components}_components_{target_layer1}_vs_{target_layer2}.pkl"
            if pca_option == True:
                print("inside outer if", flush=True)
                if all_acts2.shape[1] > 1000:
                    PCs_path2 = f"{res_path}/imagenet_val_{model_name}_{target_layer2}_{pooling}_pca_model_1000_PCs.pkl"
                    PCs2 = joblib.load(PCs_path2)
                    all_acts2 = all_acts2 @ PCs2.components_.T

                if first_projection == True:
                    save_path = f"{cca_dir}/cca_{model_name}_{num_components}_components_pca_{target_layer1}_vs_{target_layer2}.pkl"
            if os.path.exists(save_path):
                print(
                    datetime.now().strftime("%H:%M:%S"),
                    f"CCA already exists for {target_layer1} vs {target_layer2}  at {save_path}",
                    flush=True,
                )
                weights_dict = joblib.load(save_path)
                d1 = all_acts1 @ weights_dict["W1"]
                d2 = all_acts2 @ weights_dict["W2"]
                coefs_CCA = np.array(
                    [np.corrcoef(d1[:, i], d2[:, i])[0, 1] for i in range(d1.shape[1])]
                )
                layers_RSA[layer_idx1, layer_idx2] = np.mean(coefs_CCA)
                print(
                    datetime.now().strftime("%H:%M:%S"),
                    f"{target_layer1} vs {target_layer2} corr {np.round(np.mean(coefs_CCA), 3)}",
                    flush=True,
                )

            else:
                print(datetime.now().strftime("%H:%M:%S"), f"starting CCA", flush=True)
                cca = CCA(n_components=num_components, max_iter=1000)
                cca.fit(all_acts1, all_acts2)
                print(
                    datetime.now().strftime("%H:%M:%S"), f"finished CCA fit", flush=True
                )
                weights_dict = {}
                weights_dict["W1"] = (
                    cca.x_weights_
                )  # shape: (n_features1, n_components)
                weights_dict["W2"] = (
                    cca.y_weights_
                )  # shape: (n_features2, n_components)

                # 3. Project the data manually (optional, equivalent to fit_transform)
                d1 = all_acts1 @ weights_dict["W1"]
                d2 = all_acts2 @ weights_dict["W2"]
                coefs_CCA = np.array(
                    [np.corrcoef(d1[:, i], d2[:, i])[0, 1] for i in range(d1.shape[1])]
                )
                weights_dict["coefs"] = coefs_CCA
                joblib.dump(weights_dict, save_path)
                layers_RSA[layer_idx1, layer_idx2] = np.mean(coefs_CCA)
                print(
                    datetime.now().strftime("%H:%M:%S"),
                    f"{target_layer1} vs {target_layer2} corr {np.round(np.mean(coefs_CCA), 3)}",
                    flush=True,
                )
    csv_save_path = f"{cca_dir}/{model_name}_similarity_layers.csv"
    if pca_option == True:
        csv_save_path = f"{cca_dir}/{model_name}_similarity_layers_pca.csv"
    np.savetxt(csv_save_path, layers_RSA, delimiter=",")


def CCA_loop_between_mod(model_names, pooling, num_components, pca_option, res_path):
    layer_names = [get_relevant_output_layers(m) for m in model_names]
    cca_dir = f"{res_path}/cca_{model_names[0]}_vs_{model_names[1]}_{pooling}"
    os.makedirs(cca_dir, exist_ok=True)
    layers_RSA = np.zeros((len(layer_names[0]), len(layer_names[1])))
    for layer_idx1 in range(len(layer_names[0])):
        target_layer1 = layer_names[0][layer_idx1]
        feats_path1 = f"{res_path}/imagenet_val_{model_names[0]}_{target_layer1}_{pooling}_features.pkl"
        print(
            datetime.now().strftime("%H:%M:%S"),
            f"starting loading {target_layer1}",
            flush=True,
        )
        all_acts1 = joblib.load(feats_path1)
        print(
            datetime.now().strftime("%H:%M:%S"),
            f"finished loading {target_layer1}",
            flush=True,
        )
        if pca_option == True:
            first_projection = False
            if all_acts1.shape[1] > 1000:
                PCs_path1 = f"{res_path}/imagenet_val_{model_names[0]}_{target_layer1}_{pooling}_pca_model_1000_PCs.pkl"
                print(
                    datetime.now().strftime("%H:%M:%S"),
                    f"starting loading PC1",
                    flush=True,
                )
                PCs1 = joblib.load(PCs_path1)
                print(
                    datetime.now().strftime("%H:%M:%S"),
                    f"starting loading PC1",
                    flush=True,
                )
                all_acts1 = all_acts1 @ PCs1.components_.T
                print(
                    datetime.now().strftime("%H:%M:%S"),
                    f"finished backprojecting in PCs1",
                    flush=True,
                )
                first_projection = True

        for layer_idx2 in range(len(layer_names[1])):
            target_layer2 = layer_names[1][layer_idx2]
            print(
                datetime.now().strftime("%H:%M:%S"),
                f"starting layers {target_layer1} vs {target_layer2}",
            )
            feats_path2 = f"{res_path}/imagenet_val_{model_names[1]}_{target_layer2}_{pooling}_features.pkl"
            all_acts2 = joblib.load(feats_path2)
            save_path = f"{cca_dir}/cca_{model_names[0]}_vs_{model_names[1]}_{num_components}_components_{target_layer1}_vs_{target_layer2}.pkl"
            if pca_option == True:
                second_projection = False
                if all_acts2.shape[1] > 1000:
                    PCs_path2 = f"{res_path}/imagenet_val_{model_names[1]}_{target_layer2}_{pooling}_pca_model_1000_PCs.pkl"
                    PCs2 = joblib.load(PCs_path2)
                    all_acts2 = all_acts2 @ PCs2.components_.T
                    second_projection = True

                if first_projection == True or second_projection == True:
                    save_path = f"{cca_dir}/cca_{model_names[0]}_vs_{model_names[1]}_{num_components}_components_pca_{target_layer1}_vs_{target_layer2}.pkl"
            if os.path.exists(save_path):
                print(
                    datetime.now().strftime("%H:%M:%S"),
                    f"CCA already exists for {target_layer1} vs {target_layer2}  at {save_path}",
                    flush=True,
                )
                weights_dict = joblib.load(save_path)
                d1 = all_acts1 @ weights_dict["W1"]
                d2 = all_acts2 @ weights_dict["W2"]
                coefs_CCA = np.array(
                    [np.corrcoef(d1[:, i], d2[:, i])[0, 1] for i in range(d1.shape[1])]
                )
                layers_RSA[layer_idx1, layer_idx2] = np.mean(coefs_CCA)
                print(
                    datetime.now().strftime("%H:%M:%S"),
                    f"{target_layer1} vs {target_layer2} corr {np.round(np.mean(coefs_CCA), 3)}",
                    flush=True,
                )

            else:
                print(datetime.now().strftime("%H:%M:%S"), f"starting CCA", flush=True)
                try:
                    cca = CCA(
                        n_components=min(
                            num_components, all_acts1.shape[1], all_acts2.shape[1]
                        ),
                        max_iter=1000,
                    )
                    cca.fit(all_acts1, all_acts2)
                    print(
                        datetime.now().strftime("%H:%M:%S"),
                        f"finished CCA fit",
                        flush=True,
                    )
                    weights_dict = {}
                    weights_dict["W1"] = (
                        cca.x_weights_
                    )  # shape: (n_features1, n_components)
                    weights_dict["W2"] = (
                        cca.y_weights_
                    )  # shape: (n_features2, n_components)

                    # 3. Project the data manually (optional, equivalent to fit_transform)
                    d1 = all_acts1 @ weights_dict["W1"]
                    d2 = all_acts2 @ weights_dict["W2"]
                    coefs_CCA = np.array(
                        [
                            np.corrcoef(d1[:, i], d2[:, i])[0, 1]
                            for i in range(d1.shape[1])
                        ]
                    )
                    weights_dict["coefs"] = coefs_CCA
                    joblib.dump(weights_dict, save_path)
                    layers_RSA[layer_idx1, layer_idx2] = np.mean(coefs_CCA)
                    print(
                        datetime.now().strftime("%H:%M:%S"),
                        f"{target_layer1} vs {target_layer2} corr {np.round(np.mean(coefs_CCA), 3)}",
                        flush=True,
                    )
                except np.linalg.LinAlgError as e:
                    print(
                        datetime.now().strftime("%H:%M:%S"),
                        f"SVD did not converge: {e} for {target_layer1} vs {target_layer2}",
                    )
                    print("setting entry as nan")
                    layers_RSA[layer_idx1, layer_idx2] = np.nan

    csv_save_path = (
        f"{cca_dir}/{model_names[0]}_vs_{model_names[1]}_similarity_layers.csv"
    )
    if pca_option == True:
        csv_save_path = (
            f"{cca_dir}/{model_names[0]}_vs_{model_names[1]}_similarity_layers_pca.csv"
        )
    np.savetxt(csv_save_path, layers_RSA, delimiter=",")




