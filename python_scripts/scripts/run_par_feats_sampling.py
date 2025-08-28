from mpi4py import MPI
import os, yaml, sys
import torch
from torchvision import models, datasets 
from torch.utils.data import DataLoader
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from parallel.parallel_funcs import master_workers_queue, print_wise, get_perms, sample_features_core
from dim_redu_anns.utils import get_relevant_output_layers
from alignment.utils import get_usual_transform
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Incremental PCA for CNN layers")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()
    task_list = get_relevant_output_layers(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_usual_transform()
    model_cls = getattr(models, args.model_name)
    model = model_cls(pretrained=True).to(device).eval()
    imagenet_val_path = f"{paths['data_path']}/imagenet/val"
    loader = DataLoader(
        datasets.ImageFolder(imagenet_val_path, transform=transform),
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        timeout=500,
    )      
    master_workers_queue(task_list, sample_features_core, *(args.model_name, model, loader, device, paths)) 
