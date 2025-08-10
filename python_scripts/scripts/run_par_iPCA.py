# from collections import defaultdict
import os, yaml, sys, time
from torchvision import datasets, models
from torch.utils.data import DataLoader
import torch

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from dim_redu_anns.parallel import (
    parallel_setup,
    get_layers_to_extract,
    run_parallel_ipca,
)
import argparse

if __name__ == "__main__":
    comm, rank, size = parallel_setup()
    parser = argparse.ArgumentParser(description="Run Incremental PCA for CNN layers")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_components", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    # === Paths ===
    imagenet_path = f"{paths['data_path']}/imagenet"
    imagenet_val_path = os.path.join(imagenet_path, "val")
    results_path = paths["results_path"]
    remaining_layers = get_layers_to_extract(
        args.model_name, args.n_components, args.results_path
    )
    # === Transforms & Dataloader ===
    transform = get_usual_transform()
    # === Load model and loader ===
    model_cls = getattr(models, model_name)
    model = model_cls(pretrained=True).to(device).eval()
    loader = DataLoader(
        datasets.ImageFolder(imagenet_val_path, transform=transform),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        timeout=500,
    )

    run_parallel_ipca(
        paths,
        model_name=args.model_name,
        n_components=args.n_components,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
