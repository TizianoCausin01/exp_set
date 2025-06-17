import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageNet
from torchvision import transforms
import os
import sys
sys.path.append("/leonardo/home/userexternal/tcausin0/exp_set/python_scripts/src")
from sparsity_in_silico.sparsity_CNN import response_prob
#import imageio
import numpy as np
import h5py


path2res = "/leonardo_work/Sis25_piasini/tcausin/exp_set_res/silico"
imagenet_path = "/leonardo_work/Sis25_piasini/tcausin/exp_set_data/imagenet"
#alexnet = models.alexnet(pretrained=True).eval()
alexnet = models.alexnet(weights=True).eval()
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3806, 0.4242, 0.3794], std=[0.2447, 0.2732, 0.2561]
        ),  # Normalization for pretrained model
    ]
)

#reader = imageio.get_reader(path2vid)
conv_layers = [
    "conv_layer1",
    "conv_layer4",
    "conv_layer7",
    "conv_layer9",
    "conv_layer11",
]
conv_layers_idx = [1, 4, 7, 9, 11]

fc_layers = ["fc_layer2", "fc_layer5"]
fc_layers_idx = [2, 5]

output_len_conv = [193600, 139968, 64896, 43264, 43264] # FIXME work out real outdims
rand_idx_conv = []
for len_repr in output_len_conv:
    rand_idx_conv.append(
        np.random.choice(np.arange(len_repr - 1), size=len_repr // 50, replace=False)
    )

output_len_fc = [4096, 4096]
rand_idx_fc = []
for len_repr in output_len_fc:
    rand_idx_fc.append(
        np.random.choice(np.arange(len_repr - 1), size=len_repr // 50, replace=False)
            )

# %%
def wrapper_hook(layer, rand_idx):
    def hook_func(module, input, output):
        out = output.detach().half().reshape(-1)
        out = out[rand_idx]
        feats[layer].append(
            out
        )  # half makes it become float16, reshape(-1) vectorizes it

    return hook_func


hook_handle = []
for conv_idx in range(len(conv_layers_idx)):
    hook_handle.append(
        alexnet.features[conv_layers_idx[conv_idx]].register_forward_hook(
            wrapper_hook(conv_layers[conv_idx], rand_idx_conv[conv_idx])
        )
    )


for fc_idx in range(len(fc_layers_idx)):
    hook_handle.append(
        alexnet.classifier[fc_layers_idx[fc_idx]].register_forward_hook(
            wrapper_hook(fc_layers[fc_idx], rand_idx_fc[fc_idx])
        )
    )

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_set = ImageNet(root=imagenet_path, split="val", transform=transform)

val_loader = DataLoader(
    val_set,
    batch_size=1,         
    shuffle=False,         
    num_workers=1
)

feats = {
        "conv_layer1": [],
        "conv_layer4": [],
        "conv_layer7": [],
        "conv_layer9": [],
        "conv_layer11": [],
        "fc_layer2": [],
        "fc_layer5": [],
    }
with torch.no_grad():  # Disable gradient tracking for evaluation
    for inputs, labels in val_loader:
    
        outputs = alexnet(inputs)           # Forward pass
        _, preds = torch.max(outputs, 1)  # Get predicted class indices

        # Example: print first batch predictions
        print(preds)

with h5py.File(f"{path2res}/freq_alexnet.h5", "w") as f:
# Iterate over dictionary items and save them in the HDF5 file
    for key, value in feats.items():
        f.create_dataset(
            key, data=value
        )  # Create a dataset for each key-value pair
