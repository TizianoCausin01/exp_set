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

    """
    Name:
        get_usual_transform

    Description:
        Returns a standard preprocessing pipeline commonly used for 
        pretrained convolutional neural networks (e.g., ResNet, AlexNet) 
        trained on ImageNet. This includes resizing, cropping, conversion 
        to tensor, and normalization.

    Inputs:
        None

    Outputs:
        transform (torchvision.transforms.Compose):
            A composition of torchvision transforms to apply to input images.
            The transformations include:
                - Resize to 256 pixels (shorter side)
                - Center crop to 224x224 pixels
                - Convert to PyTorch tensor (values in [0,1])
                - Normalize using ImageNet mean and std:
                    mean = [0.485, 0.456, 0.406]
                    std  = [0.229, 0.224, 0.225]

    Example usage:
        transform = get_usual_transform()
        image = PIL.Image.open("example.jpg")
        input_tensor = transform(image)
    """

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

    """
    Name:
        get_maxpool_evecs

    Description:
        Given a fitted PCA object and information about the layer shape,
        this function reshapes the PCA eigenvectors (components) to match
        the original feature map shape and optionally applies spatial 
        max pooling over spatial dimensions (height and width). This is useful 
        for analyzing the dominant directions (eigenvectors) of feature activations 
        from CNN layers in a compressed but interpretable form.

    Inputs:
        data (sklearn.decomposition.PCA or similar):
            A PCA object already fit to the flattened CNN features.
            Should have the attribute `components_` with shape (n_components, prod(layer_shape)).

        layer_name (str):
            Name of the CNN layer. Determines whether to apply spatial pooling.
            If the layer is fully connected or 'avgpool', pooling is skipped.

        layer_shape (tuple of ints):
            Shape of the feature map before flattening, typically (C, H, W)
            where C is channels, H is height, W is width.

    Outputs:
        max_evecs or unflat_evecs (np.ndarray):
            If pooling is applied, returns an array of shape (n_components, C) 
            containing the maximum value across spatial dimensions for each component.
            If not pooled, returns reshaped components in shape:
                - (n_components, C, H, W) for convolutional layers
                - (n_components, C) for flat layers like 'avgpool' or 'classifier' layers.
    
    Example usage:
        pca = PCA(n_components=10).fit(feature_matrix)
        evecs = get_maxpool_evecs(pca, 'layer3', (256, 7, 7))
    """

    all_PCs_shape = (data.n_components,) + layer_shape
    evecs = data.components_
    unflat_evecs = np.reshape(evecs, all_PCs_shape)
    if layer_name == 'avgpool' or ('classifier' in layer_name):
        unflat_evecs = np.squeeze(unflat_evecs)
        return unflat_evecs # don't do anything, it's already flat
    else:
        max_evecs = np.max(unflat_evecs, axis=(2,3)) # pools the max in the feats
        return max_evecs

def sample_features(loader, feature_extractor, layer_name, batch_size, num_stim, pooling="all"):

    """
    Name:
        sample_features

    Description:
        Extracts feature activations from a specified layer of a model using a given
        `feature_extractor`. Features are sampled in batches from a DataLoader until
        a target number of stimuli is reached. Optionally, spatial pooling (max or average)
        is applied to compress spatial dimensions.

    Inputs:
        loader (torch.utils.data.DataLoader):
            DataLoader that yields batches of input images and labels.
        
        feature_extractor (nn.Module):
            A model wrapped with `torchvision.models.feature_extraction.create_feature_extractor`
            that returns activations from intermediate layers.
        
        layer_name (str):
            Name of the layer whose activations should be extracted.
        
        batch_size (int):
            The batch size used in the loader (needed to avoid over-collecting samples).
        
        num_stim (int):
            Maximum number of samples to extract. If set to 0, all available samples will be used.
        
        pooling (str, optional):
            Pooling strategy applied to the spatial dimensions of the feature maps.
            Options:
                - "all" (default): Flatten the feature maps completely.
                - "maxpool": Apply max pooling over spatial dimensions.
                - "avgpool": Apply average pooling over spatial dimensions.

    Outputs:
        all_feats (list of np.ndarray):
            A list of NumPy arrays of shape `(batch_size, feature_dim)` or similar, containing
            the extracted and optionally pooled features for each batch.

    Example usage:
        feats = sample_features(loader, extractor, "layer3", batch_size=32, num_stim=512, pooling="maxpool")
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_stim == 0:
        num_stim = len(loader.dataset)
    counter = 0
    all_feats = []
    for inputs, _ in loader:
        counter += 1
        if counter*batch_size > num_stim:
            break
        print(datetime.now().strftime("%H:%M:%S"), f"starting batch {counter}", flush=True)
        with torch.no_grad():
            inputs = inputs.to(device)
            feats = feature_extractor(inputs)[layer_name]
            if pooling== "maxpool":
                if layer_name == 'avgpool' or ('classifier' in layer_name) or (layer_name == "heads.head"):
                    feats = feats.cpu().numpy()
                else:
                    if model_name == "vit_b_16" and layer_name != "conv_proj":
                        feats = np.max(feats.cpu().numpy(), axis=0) # pools the max in the feats
                    else:
                        feats = np.max(feats.cpu().numpy(), axis=(2,3)) # pools the max in the feats
            elif pooling== "avgpool":
                if layer_name == 'avgpool' or ('classifier' in layer_name) or (layer_name == "heads.head"):
                    feats = feats.cpu().numpy()
                else:
                    if model_name == "vit_b_16" and layer_name != "conv_proj":
                        feats = np.mean(feats.cpu().numpy(), axis=0) # pools the max in the feats
                    else:
                        feats = np.mean(feats.cpu().numpy(), axis=(2,3)) # pools the max in the feats
            elif pooling == "all":
                feats = feats.view(feats.size(0), -1).cpu().numpy()
            all_feats.append(feats)
    return all_feats


def features_extraction_loop(layer_names, model_name, model, batch_size, num_images, pooling, transform, num_workers, imagenet_val_path, results_path):

    """
    Name:
        features_extraction_loop

    Description:
        Iterates through a list of specified layer names and extracts features from each using
        a pre-trained model. The extracted features from ImageNet validation images are saved
        to disk using joblib for later use.

    Inputs:
        layer_names (list of str):
            List of target layer names from which to extract features.
        
        model_name (str):
            Identifier for the model architecture (e.g., "resnet18", "alexnet").
        
        model (torch.nn.Module):
            Pretrained PyTorch model to extract features from.
        
        batch_size (int):
            Batch size for image loading and inference.
        
        num_images (int):
            Total number of images to sample from the ImageNet validation set.
        
        pooling (str):
            Pooling method to apply on the feature maps. One of:
                - "all": flatten features completely
                - "maxpool": max pool over spatial dimensions
                - "avgpool": average pool over spatial dimensions
        
        transform (torchvision.transforms.Compose):
            Image preprocessing pipeline (resize, crop, normalization, etc.).
        
        num_workers (int):
            Number of subprocesses to use for data loading.
        
        imagenet_val_path (str):
            Path to the ImageNet validation dataset directory.
        
        results_path (str):
            Directory to save the output `.pkl` files containing extracted features.

    Outputs:
        None. Features are saved to disk as `.pkl` files for each layer.

    Example usage:
        features_extraction_loop(
            layer_names=["layer3", "avgpool"],
            model_name="resnet18",
            model=model,
            batch_size=32,
            num_images=1000,
            pooling="avgpool",
            transform=get_usual_transform(),
            num_workers=4,
            imagenet_val_path="/path/to/imagenet/val",
            results_path="./features"
        )
    """

    for target_layer in layer_names:
        save_name = (
            f"imagenet_val_{model_name}_{target_layer}_{pooling}_features.pkl"
        )
        save_path = os.path.join(results_path, save_name)
        if os.path.exists(save_path):
            print(
                datetime.now().strftime("%H:%M:%S"),
                f"features already exists for {target_layer} at {save_path}",
                flush=True
            ) 
        else:
            loader = DataLoader(
            datasets.ImageFolder(imagenet_val_path, transform=transform),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            timeout=500,
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
                flush=True
            )
