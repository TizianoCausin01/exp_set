import numpy as np
from sklearn.decomposition import IncrementalPCA
import joblib
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from collections import defaultdict
from datetime import datetime
import torch
import os
from .utils import get_relevant_output_layers, worker_init_fn, get_layer_out_shape


def run_ipca_pipeline(
    model_name="resnet18",
    layers_to_extract=None,
    n_components=1000,
    batch_size=512,
    num_workers=2,
):

    """
    Name:
        run_ipca_pipeline

    Description:
        Runs an Incremental PCA (IPCA) pipeline over a specified set of layers from a pretrained 
        CNN or vision transformer model (e.g., ResNet, VGG, ViT), using features extracted from 
        the ImageNet validation set. The resulting PCA model is saved to disk for each layer.
        The function skips layers for which PCA models already exist.

    Inputs:
        model_name (str): 
            Name of the torchvision model to use (e.g., 'resnet18', 'resnet50', 'vgg16', 'alexnet', 'vit_b_16').
        layers_to_extract (list or None): 
            List of layer names to extract activations from. If None, defaults to brain-inspired layers via 
            `get_relevant_output_layers(model_name)`.
        n_components (int): 
            Maximum number of PCA components to retain (limited by number of features per layer).
        batch_size (int): 
            Batch size to use for image loading and feature extraction.
        num_workers (int): 
            Number of subprocesses to use for data loading.

    Outputs:
        None. Saves a `.pkl` file for each layer containing the fitted IncrementalPCA model to disk.
        Files are stored under:
            /leonardo_work/Sis25_piasini/tcausin/exp_set_res/silico/

    Example Usage:
        >>> run_ipca_pipeline(model_name='resnet18', n_components=512)

    Notes:
        - This function assumes access to the ImageNet validation set stored at:
            /leonardo_work/Sis25_piasini/tcausin/exp_set_data/imagenet/val
        - The function avoids recomputing PCA for layers that already have a saved model.
        - Feature extraction is done one layer at a time, looping through the dataset for each layer separately.
        - Uses `IncrementalPCA` from scikit-learn, making it scalable to large datasets.
        - The function internally applies ImageNet-standard transforms before feeding images into the model.
        - Activations are flattened (e.g., C×H×W → 1D vector) before PCA fitting.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # === Paths ===
    imagenet_path = "/leonardo_work/Sis25_piasini/tcausin/exp_set_data/imagenet"
    imagenet_val_path = os.path.join(imagenet_path, "val")
    results_path = "/leonardo_work/Sis25_piasini/tcausin/exp_set_res/silico"
    # === Transforms & Dataloader ===
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # === Load model ===
    model_cls = getattr(models, model_name)
    model = model_cls(pretrained=True).to(device).eval()
    if layers_to_extract is None:
        layers_to_extract = get_relevant_output_layers(model_name)
    # Filter out already done layers
    remaining_layers = []
    for layer in layers_to_extract:
        save_name = (
            f"imagenet_val_{model_name}_{layer}_pca_model_{n_components}_PCs.pkl"
        )
        path = os.path.join(results_path, save_name)
        if os.path.exists(path):
            print(
                datetime.now().strftime("%H:%M:%S"),
                f"PCA model already exists for {layer} → {path}",
            )
        else:
            remaining_layers.append(layer)
    if len(remaining_layers) == 0:
        print(
            datetime.now().strftime("%H:%M:%S"),
            "All PCA models already exist. Nothing to do.",
        )
        return
    print(
        datetime.now().strftime("%H:%M:%S"),
        f"Model: {model_name} | Layers to process: {len(remaining_layers)}",
    )

    # === Loop over layers separately ===
    print(datetime.now().strftime("%H:%M:%S"), "Using multiple passes (1 per layer)...")
    for layer_name in remaining_layers:
        print(
            datetime.now().strftime("%H:%M:%S"), f"Fitting PCA for layer: {layer_name}"
        )
        feature_extractor = create_feature_extractor(
            model, return_nodes=[layer_name]
        ).to(device)
        tmp_shape = get_layer_out_shape(feature_extractor, layer_name)
        n_features = np.prod(tmp_shape)  # [C, H, W] -> C*H*W
        n_components_layer = min(
            n_features, n_components
        )  # Limit to number of features
        pca = IncrementalPCA(n_components=n_components_layer)
        loader = DataLoader(
            datasets.ImageFolder(imagenet_val_path, transform=transform),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            timeout=100,
        )  # shuffle=True, took out bc I want my feats aligned
        counter = 0
        for inputs, _ in loader:
            counter += 1
            print(datetime.now().strftime("%H:%M:%S"), f"starting batch {counter}")
            with torch.no_grad():
                inputs = inputs.to(device)
                feats = feature_extractor(inputs)[layer_name]
                feats = feats.view(feats.size(0), -1).cpu().numpy()
                pca.partial_fit(feats)
        save_name = (
            f"imagenet_val_{model_name}_{layer_name}_pca_model_{n_components}_PCs.pkl"
        )
        path = os.path.join(results_path, save_name)
        joblib.dump(pca, path)
        print(
            datetime.now().strftime("%H:%M:%S"),
            f"Saved PCA for {layer_name} ?~F~R {path}",
        )


def run_ipca_pool(
    model_name="resnet18",
    pooling="maxpool",
    layers_to_extract=None,
    n_components=1000,
    batch_size=512,
    num_workers=2,
):

    """
    Name:
        run_ipca_maxpool

    Description:
        Extracts features from specified layers of a pretrained CNN on ImageNet validation images,
        applies global max pooling (if applicable), and fits an Incremental PCA model per layer.
        The PCA models are saved for later dimensionality-reduction-based analyses of model representations.

    Inputs:
        model_name (str): 
            Name of the torchvision model to use (e.g., 'resnet18', 'resnet50', 'vgg16', 'alexnet').
            Note: 'vit_b_16' is explicitly not supported.
        layers_to_extract (list or None): 
            List of layer names to extract activations from. If None, a predefined set of layers
            corresponding to brain-like stages (e.g., V1, V4, IT) is used based on the model.
        n_components (int): 
            Number of principal components to keep in PCA.
        batch_size (int): 
            Batch size used for ImageNet validation data loading.
        num_workers (int): 
            Number of subprocesses to use for data loading.

    Outputs:
        None directly returned.
        Saves fitted PCA models (as `.pkl` files) for each layer in a specified results directory.
        PCA files are named with format:
            `imagenet_val_{model_name}_{layer_name}_max_pool_pca_model_{n_components}_PCs.pkl`

    Example Usage:
        >>> run_ipca_maxpool(model_name='resnet18', n_components=500)

    Notes:
        - Uses torchvision pretrained models and ImageNet validation data.
        - For convolutional layers, global max pooling is applied before PCA.
        - Fully connected or already-pooled layers are passed through as-is.
        - Uses IncrementalPCA to process data in batches, allowing for large-scale feature extraction.
        - Skips layers for which a PCA model has already been computed and saved.
        - Assumes access to the ImageNet dataset under:
              /leonardo_work/Sis25_piasini/tcausin/exp_set_data/imagenet/val
        - Output PCA models are saved under:
              /leonardo_work/Sis25_piasini/tcausin/exp_set_res/silico
        - Custom layer mappings are defined via `get_relevant_output_layers`.
        - The helper function `get_layer_out_shape` and `worker_init_fn` must be defined elsewhere in the codebase.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # === Paths ===
    imagenet_path = "/leonardo_work/Sis25_piasini/tcausin/exp_set_data/imagenet"
    imagenet_val_path = os.path.join(imagenet_path, "val")
    results_path = "/leonardo_work/Sis25_piasini/tcausin/exp_set_res/silico"
    # === Transforms & Dataloader ===
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # === Load model ===
    model_cls = getattr(models, model_name)
    model = model_cls(pretrained=True).to(device).eval()
    if layers_to_extract is None:
        layers_to_extract = get_relevant_output_layers(model_name)
    # Filter out already done layers
    remaining_layers = []
    for layer in layers_to_extract:
        save_name = (
            f"imagenet_val_{model_name}_{layer}_maxpool_pca_model_{n_components}_PCs.pkl"
        )
        path = os.path.join(results_path, save_name)
        if os.path.exists(path):
            print(
                datetime.now().strftime("%H:%M:%S"),
                f"PCA model already exists for {layer} → {path}",
            )
        else:
            remaining_layers.append(layer)
    if len(remaining_layers) == 0:
        print(
            datetime.now().strftime("%H:%M:%S"),
            "All PCA models already exist. Nothing to do.",
        )
        return
    print(
        datetime.now().strftime("%H:%M:%S"),
        f"Model: {model_name} | Layers to process: {len(remaining_layers)}",
    )

    # === Loop over layers separately ===
    print(datetime.now().strftime("%H:%M:%S"), "Using multiple passes (1 per layer)...")
    for layer_name in remaining_layers:
        print(
            datetime.now().strftime("%H:%M:%S"), f"Fitting PCA for layer: {layer_name}"
        )
        feature_extractor = create_feature_extractor(
            model, return_nodes=[layer_name]
        ).to(device)
        tmp_shape = get_layer_out_shape(feature_extractor, layer_name)
        n_components_layer = min(
            n_features, n_components
        )  # Limit to number of features
        pca = IncrementalPCA(n_components=n_components_layer)
        loader = DataLoader(
            datasets.ImageFolder(imagenet_val_path, transform=transform),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            timeout=100,
        )  # shuffle=True, took out bc I want my feats aligned
        counter = 0
        for inputs, _ in loader:
            counter += 1
            print(datetime.now().strftime("%H:%M:%S"), f"starting batch {counter}")
            with torch.no_grad():
                inputs = inputs.to(device)
                feats = feature_extractor(inputs)[layer_name].cpu().numpy()
                if layer_name == 'avgpool' or 'classifier' in layer_name:
                    pass # don't do anything, it's already flat
                else:
                    if pooling == "maxpool":
                        feats = np.max(feats, axis=(2,3)) # pools the max in the feats
                    elif pooling == "avgpool":
                        feats = np.mean(feats, axis=(2,3))
                        
                pca.partial_fit(feats)
        
        save_name = (
            f"imagenet_val_{model_name}_{layer_name}_{pooling}_pca_model_{n_components}_PCs.pkl"
        )
        path = os.path.join(results_path, save_name)
        joblib.dump(pca, path)
        print(
            datetime.now().strftime("%H:%M:%S"),
            f"Saved PCA for {layer_name} ?~F~R {path}",
        )


def offline_ipca_pool(
    model_name="resnet18",
    pooling="maxpool",
    n_components=1000,
    batch_size=512,
    results_path="/Users/tizianocausin/OneDrive - SISSA/data_repo/exp_set_res/silico/",
):

    """
    Perform Incremental PCA (IPCA) on pooled feature activations extracted from a given model's layers,
    and save the fitted PCA model for each layer to disk.

    This function assumes that pooled features have already been computed and saved as .pkl files.

    Parameters:
    ----------
    model_name : str, default="resnet18"
        The name of the CNN model from which features were extracted.
        Used to determine the relevant layers and feature paths.

    pooling : str, default="maxpool"
        The type of pooling used when features were originally extracted ("maxpool" or "avgpool").

    n_components : int, default=1000
        The number of principal components to keep for dimensionality reduction.

    batch_size : int, default=512
        The number of feature vectors to process at once during incremental fitting.

    results_path : str
        Path to the directory containing the pooled feature files and where the PCA models will be saved.

    Outputs:
    -------
    None
        Saves fitted IncrementalPCA models (as .pkl files) to the specified `results_path`.

    Notes:
    -----
    - This function skips layers for which a PCA model already exists.
    - It loads pre-pooled features from disk (assumed to be joblib .pkl files).
    - IPCA is used to avoid loading the full dataset into memory at once.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = get_relevant_output_layers(model_name)
    for l in layers:
        features_path = f"{results_path}/imagenet_val_{model_name}_{l}_{pooling}_features.pkl"
        save_name = (
            f"imagenet_val_{model_name}_{l}_{pooling}_pca_model_{n_components}_PCs.pkl"
        )
        save_path = os.path.join(results_path, save_name)
        if os.path.exists(save_path):
            print(
                datetime.now().strftime("%H:%M:%S"),
                f"PCA model already exists for {l} in {save_path}",
            )
        else:
            feats = joblib.load(features_path)
            data_dim = feats.shape
            n_components_layer = min(n_components, data_dim[1])
            pca = IncrementalPCA(n_components=n_components_layer)
            for i_batch in range(0, data_dim[0], batch_size):
                end = min(i_batch + batch_size, data_dim[0])
                chunk = feats[i_batch:end, :]
                print(datetime.now().strftime("%H:%M:%S"), f"batch_start {i_batch} out of {data_dim[0]}", flush=True)
                pca.partial_fit(feats)
                
            joblib.dump(pca, save_path)
            print(
                datetime.now().strftime("%H:%M:%S"),
                f"Saved PCA for {l} in {save_path}",
            )


def get_top_n_dimensions(model_name, model, loader, extreme_n_imgs, top_n_PCs, num_stim, batch_size, paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = get_relevant_output_layers(model_name)
    if num_stim == 0:
        num_stim = 50000
    for target_layer in layers:
        counter = 0
        top_save_path = f"{paths["results_path"]}/{model_name}_{target_layer}_top_{extreme_n_imgs}_imgs_{top_n_PCs}_PCs.csv"
        bottom_save_path = f"{paths["results_path"]}/{model_name}_{target_layer}_bottom_{extreme_n_imgs}_imgs_{top_n_PCs}_PCs.csv"
        feature_extractor = create_feature_extractor(
        model, return_nodes=[target_layer]
        )
        if os.path.exists(top_save_path) & os.path.exists(bottom_save_path):
            print("top PCs already exist for alexnet")
        else:
            PCs_path = f"{paths["results_path"]}/imagenet_val_{model_name}_{target_layer}_pca_model_1000_PCs.pkl"
            PCs = joblib.load(PCs_path).components_
            target_PCs = PCs[:top_n_PCs]
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
                    current_dim_redu_feats = feats @ target_PCs.T
                    try:
                        all_dim_redu_feats = np.concatenate([all_dim_redu_feats, current_dim_redu_feats], axis=0)
                    except NameError:
                        all_dim_redu_feats = current_dim_redu_feats
                    # end try 
                # end with torch.no_grad():
            # end for inputs, _ in loader:
            top_n_all = []
            bottom_n_all = []
            print(datetime.now().strftime("%H:%M:%S"), f"finished passing stimuli", flush=True)
            for d in range(top_n_PCs):
                curr_dim = all_dim_redu_feats[:,d]
                idx = np.argsort(curr_dim)
                bottom_n_cd = idx[:extreme_n_imgs]
                top_n_cd = idx[extreme_n_imgs:]
                top_n_all.append(top_n_cd)
                bottom_n_all.append(bottom_n_cd)
            # end for d in range(top_n_PCs):
            top_to_save = np.stack(top_n_all, axis=0)
            bottom_to_save = np.stack(bottom_n_all, axis=0)
            np.savetxt(top_save_path, top_to_save, delimiter=",", fmt='%d')
            np.savetxt(bottom_save_path, bottom_to_save, delimiter=",", fmt='%d')
            print(datetime.now().strftime("%H:%M:%S"), f"saved files", flush=True)
        # end for target_layer in layers:
    # end if os.path.exists(top_save_path) & os.path.exists(bottom_save_path):
