from datetime import datetime 

def worker_init_fn(worker_id):
    print(datetime.now().strftime("%H:%M:%S"), f":builder: Worker {worker_id} started")

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
