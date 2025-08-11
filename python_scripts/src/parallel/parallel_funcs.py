from mpi4py import MPI
import numpy as np

from sklearn.decomposition import IncrementalPCA
from sklearn.cross_decomposition import CCA
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
from dim_redu_anns.utils import (
    get_relevant_output_layers,
    worker_init_fn,
    get_layer_out_shape,
)


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
    tot_n = len(task_list) - 1
    next_to_do = 0
    if rank == 0:
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
            if data == np.int32(-1):
                break
            func(rank, task_list[data], *args)
            comm.send(
                np.int32(1), dest=root, tag=11
            )  # Send data to process with rank 1
            print_wise(f"free again", rank=rank)

    print_wise("finished", rank=rank)
    MPI.Finalize()
# def run_parallel_ipca(
#     paths,
#     model_name="resnet18",
#     layers_to_extract=None,
#     n_components=1000,
#     batch_size=512,
#     num_workers=2,
# ):

#     from alignment.utils import get_usual_transform

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # === Paths ===
#     imagenet_path = f"{paths['data_path']}/imagenet"
#     imagenet_val_path = os.path.join(imagenet_path, "val")
#     results_path = paths["results_path"]
#     # === Transforms & Dataloader ===
#     transform = get_usual_transform()
#     # === Load model and loader ===
#     model_cls = getattr(models, model_name)
#     model = model_cls(pretrained=True).to(device).eval()
#     loader = DataLoader(
#         datasets.ImageFolder(imagenet_val_path, transform=transform),
#         batch_size=batch_size,
#         num_workers=num_workers,
#         shuffle=True,
#         pin_memory=True,
#         worker_init_fn=worker_init_fn,
#         timeout=500,
#     )
#     if layers_to_extract is None:
#         layers_to_extract = get_relevant_output_layers(model_name)
#     # Filter out already done layers
#     remaining_layers = []
#     for layer in layers_to_extract:
#         save_name = (
#             f"imagenet_val_{model_name}_{layer}_pca_model_{n_components}_PCs.pkl"
#         )
#         path = os.path.join(results_path, save_name)
#         if os.path.exists(path):
#             print(
#                 datetime.now().strftime("%H:%M:%S"),
#                 f"PCA model already exists for {layer} in {path}",
#                 flush=True,
#             )
#         else:
#             remaining_layers.append(layer)
#     if len(remaining_layers) == 0:
#         print(
#             datetime.now().strftime("%H:%M:%S"),
#             "All PCA models already exist. Nothing to do.",
#             flush=True,
#         )
#         return
#     print(
#         datetime.now().strftime("%H:%M:%S"),
#         f"Model: {model_name} | Layers to process: {len(remaining_layers)}",
#         flush=True,
#     )

#     # === Loop over layers separately ===
#     print(
#         datetime.now().strftime("%H:%M:%S"),
#         "Using multiple passes (1 per layer)...",
#         flush=True,
#     )
#     for layer_name in remaining_layers:
#         ipca_core(
#             model, model_name, layer_name, n_components, loader, results_path, device
#         )
#     # for layer_name in remaining_layers:


def ipca_core(rank, layer_name, model_name, n_components, model, loader, device, paths):

    save_name = (
        f"imagenet_val_{model_name}_{layer_name}_pca_model_{n_components}_PCs.pkl"
    )
    path = os.path.join(paths["results_path"], save_name)
    if os.path.exists(path):
        print_wise(f"{path} already exists")
    else:
        print_wise(f"Fitting PCA for layer: {layer_name}", rank=rank)
        feature_extractor = create_feature_extractor(
            model, return_nodes=[layer_name]
        ).to(device)
        tmp_shape = get_layer_out_shape(feature_extractor, layer_name)
        n_features = np.prod(tmp_shape)  # [C, H, W] -> C*H*W
        n_components_layer = min(
            n_features, n_components
        )  # Limit to number of features
        pca = IncrementalPCA(n_components=n_components_layer)
        counter = 0
        for inputs, _ in loader:
            counter += 1
            print_wise(f"starting batch {counter}", rank=rank)
            with torch.no_grad():
                inputs = inputs.to(device)
                feats = feature_extractor(inputs)[layer_name]
                feats = feats.view(feats.size(0), -1).cpu().numpy()
                pca.partial_fit(feats)

        joblib.dump(pca, path)
        print_wise(f"Saved PCA for {layer_name} at {path}", rank=rank)


def get_perms(model_names):
    all_perms = []
    layer_names = [get_relevant_output_layers(m) for m in model_names]
    for i in layer_names[0]:
        for j in layer_names[1]:
            all_perms.append([i, j])
        # end for j in layer_names[1]:
    # end for j in layer_names[1]:
    return all_perms


# EOF


def CCA_core(rank, layer_names, model_names, pooling, num_components, paths):

    cca_dir = (
        f"{paths['results_path']}/cca_{model_names[0]}_vs_{model_names[1]}_{pooling}"
    )
    target_layer1 = layer_names[0]
    target_layer2 = layer_names[1]
    os.makedirs(cca_dir, exist_ok=True)
    save_path = f"{cca_dir}/cca_{model_names[0]}_vs_{model_names[1]}_{num_components}_components_pca_{target_layer1}_vs_{target_layer2}.pkl"
    if os.path.exists(save_path):
        print_wise(
            f"CCA already exists for {target_layer1} vs {target_layer2}  at {save_path}",
            rank=rank,
        )
    else:
        print_wise(f"starting layers {target_layer1} vs {target_layer2}", rank=rank)
        feats_path1 = f"{paths['results_path']}/imagenet_val_{model_names[0]}_{target_layer1}_{pooling}_features.pkl"
        all_acts1 = joblib.load(feats_path1)
        feats_path2 = f"{paths['results_path']}/imagenet_val_{model_names[1]}_{target_layer2}_{pooling}_features.pkl"
        all_acts2 = joblib.load(feats_path2)
        print_wise(
            f"finished loading feats, size {all_acts1.shape} {all_acts2.shape} , starting CCA",
            rank=rank,
        )
        try:
            cca = CCA(
                n_components=min(
                    num_components, all_acts1.shape[1], all_acts2.shape[1]
                ),
                max_iter=1000,
            )
            cca.fit(all_acts1, all_acts2)
            print("finished CCA fitting", rank=rank)
            weights_dict = {}
            weights_dict["W1"] = cca.x_weights_  # shape: (n_features1, n_components)
            weights_dict["W2"] = cca.y_weights_  # shape: (n_features2, n_components)
            # 3. Project the data manually (optional, equivalent to fit_transform)
            d1 = all_acts1 @ weights_dict["W1"]
            d2 = all_acts2 @ weights_dict["W2"]
            coefs_CCA = np.array(
                [np.corrcoef(d1[:, i], d2[:, i])[0, 1] for i in range(d1.shape[1])]
            )
            weights_dict["coefs"] = coefs_CCA
            joblib.dump(weights_dict, save_path)
            print_wise(
                f"{target_layer1} vs {target_layer2} corr {np.round(np.mean(coefs_CCA), 3)}"
            )
        except np.linalg.LinAlgError as e:
            print_wise(
                f"SVD did not converge: {e} for {target_layer1} vs {target_layer2}",
                rank=rank,
            )


def sample_features_core(rank, layer_name, model_name, model, loader, device, paths):
    counter = 0
    save_path = f"{paths['results_path']}/imagenet_val_{model_name}_{layer_name}_PC_pool_features.pkl"
    if os.path.exists(save_path):
        print_wise(f"{save_path} already exists", rank=rank)
        return
    # end if os.path.exists(save_path):
    PCs_path = f"{paths['results_path']}/imagenet_val_{model_name}_{layer_name}_pca_model_1000_PCs.pkl"
    PCs = joblib.load(PCs_path).components_
    feature_extractor = create_feature_extractor(model, return_nodes=[layer_name]).to(
        device
    )
    all_feats = []
    for inputs, _ in loader:
        counter += 1
        print_wise(f"starting batch {counter}", rank=rank)
        with torch.no_grad():
            inputs = inputs.to(device)
            feats = feature_extractor(inputs)[layer_name]
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            feats = feats @ PCs.T
            all_feats.append(feats)
    # end for inputs, _ in loader:
    all_acts = np.concatenate(all_feats, axis=0)
    joblib.dump(all_acts, save_path)
    print_wise(f"Saved features for {layer_name} at {save_path}", rank=rank)


# EOF
