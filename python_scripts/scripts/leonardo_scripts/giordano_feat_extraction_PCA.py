import torch
import torch.nn.functional as F
import os
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import joblib
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from collections import defaultdict
from datetime import datetime

def worker_init_fn(worker_id):
    print(datetime.now().strftime("%H:%M:%S"), f":builder: Worker {worker_id} started")

# Helper to get candidate layers for PCA
def get_relevant_output_layers(model, model_name):
    """
    Returns a list of brain-relevant layers to extract from a given model,
    approximately mapping to V1, V4, and IT.
    """
    if model_name == 'resnet18':
        return [
            'conv1',                         # V1
            'layer1.0.relu_1',               # V2
            'layer1.1.relu_1',               # V2/V4
            'layer2.0.relu_1',               # V4
            'layer2.1.relu_1',               # V4
            'layer3.0.relu_1',               # V4/IT
            'layer3.1.relu_1',               # IT
            'layer4.0.relu_1',               # IT
            'layer4.1.relu_1',               # IT
            'avgpool'                        # pooled features (IT-like)
        ]
    if model_name == 'resnet50':
        return [
            'conv1',                         # V1
            'layer1.0.relu_2',
            'layer1.1.relu_2',               # V2
            'layer1.2.relu_2',               # V2
            'layer2.0.relu_2',
            'layer2.1.relu_2',               # V4
            'layer2.2.relu_2',               # V4
            'layer2.3.relu_2',               # V4
            'layer3.0.relu_2',
            'layer3.1.relu_2',               # V4/IT
            'layer3.2.relu_2',               # V4/IT
            'layer3.3.relu_2',               # IT-like
            'layer3.4.relu_2',
            'layer3.5.relu_2',               # IT-like
            'layer4.0.relu_2',
            'layer4.1.relu_2',               # IT-like
            'layer4.2.relu_2',
            'avgpool'
        ]
    if model_name == 'vgg16':
        return [
            'features.0',       # conv1_1 (V1)
            'features.2',       # conv1_2
            'features.5',       # conv2_2
            'features.10',      # conv3_3
            'features.12',      # conv4_1
            'features.16',      # conv4_3
            'features.19',      # conv5_1
            'features.23',      # conv5_3
            'features.30',      # final conv
            'classifier.0'      # first FC layer
        ]
    if model_name == 'alexnet':
        return [
            'features.0',       # conv1
            'features.4',       # conv2
            'features.7',       # conv3
            'features.9',       # conv4
            'features.11',      # conv5
            'classifier.2',     # fc6
            'classifier.5'      # fc7
        ]
    if model_name == 'vit_b_16':
        return [
            'conv_proj',                                      # patch embedding (V1-like)
            'encoder.layers.encoder_layer_0.add_1',           # early transformer block ← V1
            'encoder.layers.encoder_layer_2.add_1',           # mid/early block
            'encoder.layers.encoder_layer_4.add_1',           # mid
            'encoder.layers.encoder_layer_6.add_1',           # V4-like
            'encoder.layers.encoder_layer_8.add_1',           # higher block
            'encoder.layers.encoder_layer_10.add_1',          # deep
            'encoder.layers.encoder_layer_11.add_1',          # very deep ← IT
            'encoder.ln',                                     # final transformer output
            'heads.head'                                      # classification head ← IT
        ]
    raise ValueError(f"Model {model_name} not supported in `get_relevant_output_layers()`.")
    # else:
    #     all_nodes, _ = get_graph_node_names(model)
    #     # Keep layers that are ReLU outputs or last ReLU in a residual block
    #     return [name for name in all_nodes if name.endswith('relu_1') or name.endswith('relu')]


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
        save_name = f"imagenet_val_{model_name}_{layer}_pca_model.pkl"
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
            timeout=100
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
            save_name = f"imagenet_val_{model_name}_{layer_name}_pca_model.pkl"
            path = os.path.join(results_path, save_name)
            joblib.dump(pca, path)
            print(datetime.now().strftime("%H:%M:%S"), f"Saved PCA for {layer_name} → {path}")
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
            for inputs in loader:
                counter+=1
                print(f"starting batch {counter}")
                with torch.no_grad():
                    inputs = inputs.to(device)
                    feats = feature_extractor(inputs)[layer_name]
                    feats = feats.view(feats.size(0), -1).cpu().numpy()
                    pca.partial_fit(feats)
            save_name = f"imagenet_val_{model_name}_{layer_name}_pca_model.pkl"
            path = os.path.join(results_path, save_name)
            joblib.dump(pca, path)
            print(datetime.now().strftime("%H:%M:%S"), f"Saved PCA for {layer_name} → {path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run Incremental PCA for CNN layers")
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_components', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--multiple_passes', action='store_true')  # Flag → default False
    args = parser.parse_args()
    run_pca_pipeline(
        model_name=args.model_name,
        n_components=args.n_components,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        multiple_passes=args.multiple_passes
    )
