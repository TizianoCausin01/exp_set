import numpy as np
import resource, platform
from sklearn.decomposition import PCA
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

def run_pca_pipeline(
    paths,
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
    from alignment.utils import get_usual_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # === Paths ===
    imagenet_path = f"{paths['data_path']}/imagenet"
    imagenet_val_path = os.path.join(imagenet_path, "val")
    results_path = paths['results_path']
    # === Transforms & Dataloader ===
    transform = get_usual_transform()
    # === Load model ===
    model_cls = getattr(models, model_name)
    model = model_cls(pretrained=True).to(device).eval()
    if layers_to_extract is None:
        layers_to_extract = get_relevant_output_layers(model_name)
    # Filter out already done layers
    remaining_layers = []
    for layer in layers_to_extract:
        save_name = (
            f"imagenet_val_{model_name}_{layer}_true_pca_model_{n_components}_PCs.pkl"
        )
        path = os.path.join(results_path, save_name)
        if os.path.exists(path):
            print(
                datetime.now().strftime("%H:%M:%S"),
                f"PCA model already exists for {layer} in {path}",
                flush=True,
            )
        else:
            remaining_layers.append(layer)
    if len(remaining_layers) == 0:
        print(
            datetime.now().strftime("%H:%M:%S"),
            "All PCA models already exist. Nothing to do.",
            flush=True,
        )
        return
    print(
        datetime.now().strftime("%H:%M:%S"),
        f"Model: {model_name} | Layers to process: {len(remaining_layers)}",
        flush=True,
    )

    # === Loop over layers separately ===
    print(datetime.now().strftime("%H:%M:%S"), "Using multiple passes (1 per layer)...", flush=True)
    for layer_name in remaining_layers:
        print(
            datetime.now().strftime("%H:%M:%S"), f"Fitting PCA for layer: {layer_name}", flush=True
        )
        feature_extractor = create_feature_extractor(
            model, return_nodes=[layer_name]
        ).to(device)
        tmp_shape = get_layer_out_shape(feature_extractor, layer_name)
        n_features = np.prod(tmp_shape)  # [C, H, W] -> C*H*W
        n_components_layer = min(
            n_features, n_components
        )  # Limit to number of features
        pca = PCA(n_components=n_components_layer)
        loader = DataLoader(
            datasets.ImageFolder(imagenet_val_path, transform=transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            timeout=500,
        )  # shuffle=True, took out bc I want my feats aligned
        counter = 0
        tot_feats = []
        for inputs, _ in loader:
            counter += 1
            max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if platform.system() == "Darwin":  # macOS
                max_rss_mb = max_rss_kb / (1024 * 1024)
            else:  # Linux
                max_rss_mb = max_rss_kb / 1024
            with torch.no_grad():
                inputs = inputs.to(device)
                feats = feature_extractor(inputs)[layer_name]
                feats = feats.view(feats.size(0), -1).cpu().numpy()
                feats = feats.astype(np.float16)
                tot_feats.append(feats)
            print(datetime.now().strftime("%H:%M:%S"), f"finished batch {counter}, data matrix len {len(tot_feats)}, size of the batch {feats.nbytes/10**9} Gb, max mem by now {max_rss_mb} Mb", flush=True)
        #end for inputs, _ in loader:
        tot_feats = np.vstack(tot_feats)
        max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":  # macOS
            max_rss_mb = max_rss_kb / (1024 * 1024)
        else:  # Linux
            max_rss_mb = max_rss_kb / 1024
        print(datetime.now().strftime("%H:%M:%S"), f"before fitting PCA for {layer_name}, data matrix shape {tot_feats.shape}, max mem by now {max_rss_mb} Mb", flush=True)
        pca.fit(tot_feats)
        max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":  # macOS
            max_rss_mb = max_rss_kb / (1024 * 1024)
        else:  # Linux
            max_rss_mb = max_rss_kb / 1024
        print(datetime.now().strftime("%H:%M:%S"), f"after fitting PCA for {layer_name}, max mem by now {max_rss_mb} Mb", flush=True)
        save_name = (
            f"imagenet_val_{model_name}_{layer_name}_true_pca_model_{n_components}_PCs.pkl" 
        )
        path = os.path.join(results_path, save_name)
        joblib.dump(pca, path)
        print(
            datetime.now().strftime("%H:%M:%S"),
            f"Saved PCA for {layer_name} at {path}", 
            flush=True,
        )


