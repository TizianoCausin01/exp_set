from mpi4py import MPI
import numpy as np
from sklearn.decomposition import IncrementalPCA
import joblib
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from collections import defaultdict
from datetime import datetime
import torch
import os
from dim_redu_anns.utils import get_relevant_output_layers, worker_init_fn, get_layer_out_shape

def print_wise(mex, rank=None):
    if rank == None:
        print(datetime.now().strftime("%H:%M:%S"), f"- {mex}", flush=True)
    else:
        print(
            datetime.now().strftime("%H:%M:%S"),
            f"- rank {rank}",
            f"{mex}",
            flush=True,
        )


# EOF



def parallel_setup():
    comm = MPI.COMM_WORLD  # Get the global communicator
    rank = comm.Get_rank()  # Get the rank (ID) of the current process
    size = comm.Get_size()  # Get the total number of processes
    return comm, rank, size


def master_workers_queue(task_list, func, *args, **kwargs):
    comm, rank, size = parallel_setup()
    root = 0
    tot_n = len(task_list) -1
    next_to_do = 0
    if rank == 0:
        # TODO check these passages with the indices... they're tricky
        for dst in range(1, size):
            comm.send(
                np.int32(next_to_do), dest=dst, tag=11
            )  # Send data to process with rank 1
            next_to_do += 1
            print_wise(f"computed {next_to_do}", rank=rank)
            if next_to_do == tot_n:
                break
            # end if done_by_now+1 > tot_n:

        # spotlight = np.zeros(size - 1)  # one means the process is free
        while next_to_do < tot_n:
            status = MPI.Status()
            d = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            src = status.Get_source()
            tag = status.Get_tag()
            comm.send(
                np.int32(next_to_do), dest=src, tag=11
            )  # Send data to process with rank 1
            next_to_do += 1
            print_wise(f"received from {src} , root : {next_to_do}", rank=rank)
        for i in range(1, size):
            comm.send(np.int32(-1), dest=i, tag=11)  # Send data to process with rank 1

    else:
        while True:
            data = comm.recv(source=0, tag=11)  # Receive data from process with rank 0
            print_wise(f"received: {data}", rank=rank)
            #func(*args, **kwargs)
            func(task_list, data)
            if data == np.int32(-1):
                break
            comm.send(
                np.int32(1), dest=root, tag=11
            )  # Send data to process with rank 1
            print_wise(f"free again", rank=rank)

    print_wise("finished", rank=rank)


def run_parallel_ipca(
    paths,
    model_name="resnet18",
    layers_to_extract=None,
    n_components=1000,
    batch_size=512,
    num_workers=2,
):

    from alignment.utils import get_usual_transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # === Paths ===
    imagenet_path = f"{paths['data_path']}/imagenet"
    imagenet_val_path = os.path.join(imagenet_path, "val")
    results_path = paths["results_path"]
    # === Transforms & Dataloader ===
    transform = get_usual_transform()
    # === Load model and loader ===
    model_cls = getattr(models, model_name)
    model = model_cls(pretrained=True).to(device).eval()
    loader = DataLoader(
        datasets.ImageFolder(imagenet_val_path, transform=transform),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        timeout=500,
    )
    if layers_to_extract is None:
        layers_to_extract = get_relevant_output_layers(model_name)
    # Filter out already done layers
    remaining_layers = []
    for layer in layers_to_extract:
        save_name = (
            f"imagenet_val_{model_name}_{layer}_pca_model_{n_components}_PCs.pkl"
        )
        path = os.path.join(results_path, save_name)
        if os.path.exists(path):
            print(
                datetime.now().strftime("%H:%M:%S"),
                f"PCA model already exists for {layer} in {path}",
                flush=True,
            )
        else:
            remaining_layers.append(layer)
    if len(remaining_layers) == 0:
        print(
            datetime.now().strftime("%H:%M:%S"),
            "All PCA models already exist. Nothing to do.",
            flush=True,
        )
        return
    print(
        datetime.now().strftime("%H:%M:%S"),
        f"Model: {model_name} | Layers to process: {len(remaining_layers)}",
        flush=True,
    )

    # === Loop over layers separately ===
    print(
        datetime.now().strftime("%H:%M:%S"),
        "Using multiple passes (1 per layer)...",
        flush=True,
    )
    for layer_name in remaining_layers:
        ipca_core(
            model, model_name, layer_name, n_components, loader, results_path, device
        )
    # for layer_name in remaining_layers:


def ipca_core(
    model, model_name, layer_name, n_components, loader, results_path, rank, device
):
    print(
        datetime.now().strftime("%H:%M:%S"),
        f"rank {rank} Fitting PCA for layer: {layer_name}",
        flush=True,
    )
    feature_extractor = create_feature_extractor(model, return_nodes=[layer_name]).to(
        device
    )
    tmp_shape = get_layer_out_shape(feature_extractor, layer_name)
    n_features = np.prod(tmp_shape)  # [C, H, W] -> C*H*W
    n_components_layer = min(n_features, n_components)  # Limit to number of features
    pca = IncrementalPCA(n_components=n_components_layer)

    counter = 0
    for inputs, _ in loader:
        counter += 1
        print(
            datetime.now().strftime("%H:%M:%S"),
            f"{rank} starting batch {counter}",
            flush=True,
        )
        with torch.no_grad():
            inputs = inputs.to(device)
            feats = feature_extractor(inputs)[layer_name]
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            pca.partial_fit(feats)
    save_name = (
        f"imagenet_val_{model_name}_{layer_name}_pca_model_{n_components}_PCs.pkl"
    )
    path = os.path.join(results_path, save_name)
    joblib.dump(pca, path)
    print(
        datetime.now().strftime("%H:%M:%S"),
        f"Saved PCA for {layer_name} at {path}",
        flush=True,
    )
