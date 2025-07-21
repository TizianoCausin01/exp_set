import os, sys, time
from matplotlib import pyplot as plt
import numpy as np
import pickle
import pandas
import gzip
import joblib
from datetime import datetime
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
#sys.path.append("/Users/tizianocausin/Desktop/backUp20240609/summer2025/ponce_lab/exp_set/python_scripts/src")
sys.path.append("/leonardo/home/userexternal/tcausin0/exp_set/python_scripts/src")
from dim_redu_anns.utils import get_relevant_output_layers
from alignment.utils import sample_features, get_usual_transform, features_extraction_loop
import argparse

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--model_name', type=str, default='resnet18')
        parser.add_argument('--num_images', type=int, default=0)
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--pooling', type=str, default="all")
        parser.add_argument('--mobilenet_opt', type=int, default=0)
        args = parser.parse_args()
        imagenet_path = "/leonardo_work/Sis25_piasini/tcausin/exp_set_data/imagenet"
        imagenet_val_path = os.path.join(imagenet_path, "val")
        results_path = "/leonardo_work/Sis25_piasini/tcausin/exp_set_res/silico"
        layer_names = get_relevant_output_layers(args.model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = get_usual_transform()
        if args.mobilenet_opt == 0:
            model_cls = getattr(models, args.model_name)
            model = model_cls(pretrained=True).to(device).eval()
        else:
            from torchvision.models import MobileNet_V3_Large_Weights
            model_name_ = "mobilenet_v3_large"
            model_cls = getattr(models, model_name_)
            if args.mobilenet_opt == 1:
                model = model_cls(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            elif args.mobilenet_opt == 2: 
                model = model_cls(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            else:
                raise ValueError("mobilenet_opt not supported")
        features_extraction_loop(layer_names, args.model_name, model, args.batch_size, args.num_images, args.pooling, transform, args.num_workers, imagenet_val_path, results_path)
