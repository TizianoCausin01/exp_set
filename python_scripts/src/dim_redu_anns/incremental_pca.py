import numpy as np
from sklearn.decomposition import IncrementalPCA
import joblib
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from collections import defaultdict
from datetime import datetime
import torch
import os
from .utils import get_relevant_output_layers, worker_init_fn

def run_pca_pipeline(model_name='resnet18', layers_to_extract=None, n_components=1000,
                     batch_size=512, multiple_passes=False, num_workers=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # === Paths ===
    imagenet_path = "/leonardo_work/Sis25_piasini/tcausin/exp_set_data/imagenet"
    alexnet = models.alexnet(weights=True).eval()
    imagenet_val_path = os.path.join(imagenet_path, "val")
    results_path = "/leonardo_work/Sis25_piasini/tcausin/exp_set_res/silico" 
    # === Transforms & Dataloader ===
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # === Load model ===
    model_cls = getattr(models, model_name)
    model = model_cls(pretrained=True).to(device).eval()
    if layers_to_extract is None:
        layers_to_extract = get_relevant_output_layers(model, model_name)
    # Filter out already done layers
    remaining_layers = []
    for layer in layers_to_extract:
        save_name = f"imagenet_val_{model_name}_{layer}_pca_model_{n_components}_PCs.pkl"
        path = os.path.join(results_path, save_name)
        if os.path.exists(path):
            print(datetime.now().strftime("%H:%M:%S"), f"PCA model already exists for {layer} → {path}")
        else:
            remaining_layers.append(layer)
    if len(remaining_layers) == 0:
        print(datetime.now().strftime("%H:%M:%S"), "All PCA models already exist. Nothing to do.")
        return
    print(datetime.now().strftime("%H:%M:%S"),f"Model: {model_name} | Layers to process: {len(remaining_layers)}")
    # === Option 1: All layers in one pass ===
    if not multiple_passes:
        feature_extractor = create_feature_extractor(model, return_nodes=remaining_layers).to(device)
        # Initialize PCA for each layer separately
        pcas = {}
        for layer in remaining_layers:
            with torch.no_grad():
                tmp_shape = feature_extractor(torch.randn(1, 3, 224, 224).to(device))[layer].shape[1:]
            n_features = np.prod(tmp_shape)  # [C, H, W] -> C*H*W
            print(datetime.now().strftime("%H:%M:%S"), f"Layer: {layer} | Number of features: {n_features}")
            n_components_layer = min(n_features, n_components)  # Limit to number of features
            pcas[layer] = IncrementalPCA(n_components=n_components_layer)
        loader = DataLoader(
            datasets.ImageFolder(imagenet_val_path, transform=transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            timeout=400
        )
        print(datetime.now().strftime("%H:%M:%S"), "Single-pass PCA fitting across all layers...")
        for inputs, _ in tqdm(loader, desc="Fitting PCA"):
            with torch.no_grad():
                inputs = inputs.to(device)
                outputs = feature_extractor(inputs)
            for layer_name, tensor in outputs.items():
                feats = tensor.view(tensor.size(0), -1).cpu().numpy()
                pcas[layer_name].partial_fit(feats)
        for layer_name, pca in pcas.items():
            save_name = f"imagenet_val_{model_name}_{layer_name}_pca_model_{n_components}_PCs.pkl"
            path = os.path.join(results_path, save_name)
            joblib.dump(pca, path)
            print(datetime.now().strftime("%H:%M:%S"), f"Saved PCA for {layer_name} ?~F~R {path}")
    # === Option 2: Loop over layers separately ===
    else:
        print(datetime.now().strftime("%H:%M:%S"), "Using multiple passes (1 per layer)...")
        for layer_name in remaining_layers:
            print(datetime.now().strftime("%H:%M:%S"), f"Fitting PCA for layer: {layer_name}")
            feature_extractor = create_feature_extractor(model, return_nodes=[layer_name]).to(device)
            with torch.no_grad():
                tmp_shape = feature_extractor(torch.randn(1, 3, 224, 224).to(device))[layer_name].shape[1:]
            n_features = np.prod(tmp_shape)  # [C, H, W] -> C*H*W
            n_components_layer = min(n_features, n_components)  # Limit to number of features
            pca = IncrementalPCA(n_components=n_components_layer)
            loader = DataLoader(
                datasets.ImageFolder(imagenet_val_path, transform=transform),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
                timeout=100
            ) # shuffle=True, took out bc I want my feats aligned
            counter = 0
            for inputs, _ in loader:
                counter+=1
                print(datetime.now().strftime("%H:%M:%S"), f"starting batch {counter}")
                with torch.no_grad():
                    inputs = inputs.to(device)
                    feats = feature_extractor(inputs)[layer_name]
                    feats = feats.view(feats.size(0), -1).cpu().numpy()
                    pca.partial_fit(feats)
            save_name = f"imagenet_val_{model_name}_{layer_name}_pca_model_{n_components}_PCs.pkl"
            path = os.path.join(results_path, save_name)
            joblib.dump(pca, path)
            print(datetime.now().strftime("%H:%M:%S"), f"Saved PCA for {layer_name} ?~F~R {path}")
