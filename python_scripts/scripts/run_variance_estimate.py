# to run like this e.g. 
# e.g. python run_running_max.py --model_name alexnet --extreme_n_imgs 10 --top_n_PCs 3 --num_stim 10 --batch_size 5 --num_workers 1

import numpy as np
import joblib
import matplotlib.pyplot as plt
import os, yaml, sys
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
import torch
from torch.utils.data import DataLoader
from alignment.utils import get_usual_transform
from dim_redu_anns.utils import get_relevant_output_layers
from dim_redu_anns.incremental_pca import get_top_n_dimensions
from torchvision import transforms, datasets, models
from datetime import datetime
import argparse
from dim_redu_anns.utils import get_relevant_output_layers, get_layer_out_shape
from sampling.utils import sum_for_var, variance, variance_estimation_loop

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--model_name', type=str, default='resnet18')
        parser.add_argument('--num_stim', type=int, default=0)
        parser.add_argument('--batch_size', type=int, default=1024)
        parser.add_argument('--num_workers', type=int, default=1)
        args = parser.parse_args()
        
        imagenet_val_path = f"{paths['data_path']}/imagenet/val"
        transform = get_usual_transform()
        
        loader = DataLoader(
            datasets.ImageFolder(imagenet_val_path, transform=transform),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            timeout=500,
        )  # shuffle=True, took out bc I want my feats aligned
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_cls = getattr(models, args.model_name)
        model = model_cls(pretrained=True).to(device).eval()
        model_name = args.model_name
        layers = get_relevant_output_layers(model_name)
        for target_layer in layers:
            feature_extractor = create_feature_extractor(
                model, return_nodes=[target_layer]
            )
            v = variance_estimation_loop(feature_extractor, target_layer, loader, args.num_stim, args.batch_size)
            path_to_save = f"{paths['results_path']}/estimated_var_{model_name}_{target_layer}.csv"
            np.savetxt(path_to_save, v, delimiter=",")
            print(datetime.now().strftime("%H:%M:%S"), f"{target_layer} saved at {path_to_save}")

        # for target_layer in layers:

