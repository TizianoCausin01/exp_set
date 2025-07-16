import os
import numpy as np 
from datetime import datetime
import joblib
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)


def get_usual_transform():
    transform = transforms.Compose(
        [       
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]           
    )
    return transform


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

def sample_features(loader, feature_extractor, layer_name, batch_size, num_stim, pooling="all"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_stim == 0:
        num_stim = len(loader.dataset)
    counter = 0
    all_feats = []
    for inputs, _ in loader:
        counter += 1
        if counter*batch_size > num_stim:
            break
        print(datetime.now().strftime("%H:%M:%S"), f"starting batch {counter}")
        with torch.no_grad():
            inputs = inputs.to(device)
            feats = feature_extractor(inputs)[layer_name]
            if pooling== "maxpool":
                if layer_name == 'avgpool' or 'classifier' in layer_name:
                    pass # don't do anything, it's already flat
                else:
                    feats = np.max(feats.cpu().numpy(), axis=(2,3)) # pools the max in the feats
            elif pooling== "avgpool":
                if layer_name == 'avgpool' or 'classifier' in layer_name:
                    pass # don't do anything, it's already flat
                else:
                    feats = np.mean(feats.cpu().numpy(), axis=(2,3)) # pools the max in the feats
            elif pooling == "all":
                feats = feats.view(feats.size(0), -1).cpu().numpy()
            all_feats.append(feats)
    return all_feats


def features_extraction_loop(layer_names, model_name, model, batch_size, num_images, pooling, transform, num_workers, imagenet_val_path, results_path):
    for target_layer in layer_names:
        save_name = (
            f"imagenet_val_{model_name}_{target_layer}_{pooling}_features.pkl"
        )
        save_path = os.path.join(results_path, save_name)
        if os.path.exists(save_path):
            print(
                datetime.now().strftime("%H:%M:%S"),
                f"PCA model already exists for {target_layer} at {save_path}",
            ) 
        else:
            loader = DataLoader(
            datasets.ImageFolder(imagenet_val_path, transform=transform),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            timeout=100,
            )  # shuffle=True, took out bc I want my feats aligned
            
            feature_extractor = create_feature_extractor(
                model, return_nodes=[target_layer]
            )
            all_feats = sample_features(loader, feature_extractor, target_layer, batch_size, num_images, pooling)
            all_acts = np.concatenate(all_feats, axis=0)
            joblib.dump(all_acts, save_path)
            print(
                datetime.now().strftime("%H:%M:%S"),
                f"Saved features for {target_layer} at {save_path}",
            )
