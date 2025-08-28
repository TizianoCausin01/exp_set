from mpi4py import MPI
import os, yaml, sys

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from parallel.parallel_funcs import (
    master_workers_queue,
    print_wise,
    get_perms,
    CCA_core,
)
from dim_redu_anns.utils import get_relevant_output_layers
from parallel.parallel_funcs import (
    parallel_setup,
    print_wise,
    setup_full_dataloader,
    perm2idx,
    batch_cka_core,
    master_merger_queue,
    diag_perms,
)
from alignment.utils import get_usual_transform
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, models
from torchvision.models import MobileNet_V3_Large_Weights
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Run Incremental PCA for CNN layers")
    parser.add_argument("--n_batches", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gram_or_cov", type=str)
    args = parser.parse_args()
    model_name_ = "mobilenet_v3_large"
    model_cls = getattr(models, model_name_)
    model1 = model_cls(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1).to(device).eval()
    model2 = model_cls(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2).to(device).eval()
    model_names = [ "mobilenet_v3_large", "mobilenet_v3_large"]
    task_list = diag_perms(model_name_)
    transform = get_usual_transform()
    loader = setup_full_dataloader(args.batch_size, paths)
    master_merger_queue(
        task_list,
        paths,
        batch_cka_core,
        *(
            model_names,
            model1,
            model2,
            loader,
            args.n_batches,
            args.gram_or_cov,
            device,
            paths,
        ),
    )
