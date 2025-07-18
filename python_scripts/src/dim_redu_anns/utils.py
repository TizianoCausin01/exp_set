from datetime import datetime 
import torch 
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from torchvision import transforms, datasets, models

import torch 
from torchvision.models.feature_extraction import (
            create_feature_extractor,
                get_graph_node_names,
                )
from torchvision import transforms, datasets, models


def worker_init_fn(worker_id):
    print(datetime.now().strftime("%H:%M:%S"), f":builder: Worker {worker_id} started")

def get_relevant_output_layers(model_name):
    """
    Name:
        get_relevant_output_layers

    Description:
        Returns a list of layer names from a specified deep neural network model
        that are approximately aligned with regions in the primate visual cortex
        — namely V1, V4, and IT (inferotemporal cortex). These layers are selected
        to enable brain-model comparisons or neuroscientific analyses of model representations.

    Inputs:
        model_name (str): 
            The name of the model architecture. Supported models include:
            - 'resnet18'
            - 'resnet50'
            - 'vgg16'
            - 'alexnet'
            - 'vit_b_16'

    Outputs:
        List[str]: 
            A list of strings representing layer names in the model. These layers are chosen
            based on their approximate correspondence to stages in the visual processing hierarchy
            (e.g., early visual cortex V1, intermediate V4, and higher-level IT).

    Example Usage:
        >>> layers = get_relevant_output_layers('resnet18')
        >>> print(layers)
        ['conv1', 'layer1.0.relu_1', 'layer1.1.relu_1', ..., 'avgpool']

        >>> layers = get_relevant_output_layers('vit_b_16')
        >>> print(layers)
        ['conv_proj', 'encoder.layers.encoder_layer_0.add_1', ..., 'heads.head']

    Notes:
        - The selected layers are meant to facilitate comparisons between model activations
          and neural recordings across the ventral stream.
        - If an unsupported model name is passed, the function raises a `ValueError`.
        - These layer names correspond to PyTorch model definitions and are used for
          feature extraction via tools like `torchvision.models.feature_extraction.create_feature_extractor`.
        - For unlisted models, consider manually inspecting the model graph using
          `torchvision.models.feature_extraction.get_graph_node_names(model)`.
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


def get_layer_out_shape(feature_extractor, layer_name):

    """
    Name:
        get_layer_out_shape

    Description:
        Computes the output shape (excluding batch size) of a specific layer 
        from a given PyTorch feature extractor when applied to a dummy input 
        image of size (1, 3, 224, 224).

    Inputs:
        feature_extractor (torch.nn.Module):
            A PyTorch model (typically a feature extractor created via
            torchvision.models.feature_extraction.create_feature_extractor)
            which outputs a dictionary of intermediate activations.
        
        layer_name (str):
            The name of the layer for which the output shape is desired. 
            This must be one of the keys returned by the feature_extractor.

    Outputs:
        tuple:
            A tuple representing the shape of the output tensor from the 
            specified layer, excluding the batch dimension. For example,
            (512, 7, 7) for a convolutional layer or (768,) for a transformer block.

    Example Usage:
        >>> from torchvision.models import resnet18
        >>> from torchvision.models.feature_extraction import create_feature_extractor
        >>> model = resnet18(pretrained=True).eval()
        >>> feat_ext = create_feature_extractor(model, return_nodes=["layer1.0.relu_1"])
        >>> shape = get_layer_out_shape(feat_ext, "layer1.0.relu_1")
        >>> print(shape)
        (64, 56, 56)

    Notes:
        - This function uses a dummy input of size (1, 3, 224, 224), which is 
          standard for ImageNet models.
        - It runs in no_grad() mode to prevent gradient tracking and reduce memory use.
        - Ensure the `layer_name` matches exactly the name used in the return_nodes 
          dictionary when creating the feature extractor.
    """

    with torch.no_grad():
        tmp_shape = feature_extractor(torch.randn(1, 3, 224, 224))[
            layer_name
        ].shape [1:]
    return tmp_shape


