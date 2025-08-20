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
)
from alignment.utils import get_usual_transform
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, models

# TODO create a function that takes as input the layers and returns the position and then create cka core

import torch.distributed as dist
from torch.utils.data import DataLoader
import torch


def master_merger_queue(task_list, paths, func, *args, **kwargs):
    comm, rank, size = parallel_setup()
    root = 0
    merger = 1
    tot_n = len(task_list)
    next_to_do = 0
    if rank == 0:
        for dst in range(2, size):
            print_wise(f"sending stuff", rank=rank)
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
        for i in range(2, size):
            comm.send(np.int32(-1), dest=i, tag=11)  # Send data to process with rank 1

    elif rank == 1:
        model_names = args[0]
        paths = args[7]
        n_batches = args[4]
        w, h = len(get_relevant_output_layers(model_names[0])), len(
            get_relevant_output_layers(model_names[1])
        )  # n layers of model_names[0] and model_names[1]
        cka_mat = np.zeros((w, h))

        counter = 0
        while counter < tot_n:
            status = MPI.Status()
            d = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            counter += 1
            print_wise(f"received {d}, {counter} tasks already processed", rank=rank)
            cka_mat[d[0], d[1]] = d[2]
            print(cka_mat)
        csv_save_path = f"{paths['results_path']}/cka_{model_names[0]}_{model_names[1]}_{n_batches}_batches.csv"
        np.savetxt(csv_save_path, cka_mat, delimiter=",")

    else:
        model_names = args[0]
        while True:
            data = comm.recv(source=0, tag=11)  # Receive data from process with rank 0
            print_wise(f"received: {data}", rank=rank)
            if data == np.int32(-1):
                break
            print_wise(f"starting cka...", rank=rank)
            res = func(rank, task_list[data], *args)
            to_send = perm2idx(rank, task_list[data], model_names)
            to_send.append(res)
            comm.send(to_send, dest=merger, tag=11)  # Send data to process with rank 1
            comm.send(
                np.int32(1), dest=root, tag=11
            )  # Send data to process with rank 1
            print_wise(f"free again", rank=rank)

    print_wise("finished", rank=rank)
    MPI.Finalize()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Run Incremental PCA for CNN layers")
    parser.add_argument("--model_name1", type=str)
    parser.add_argument("--model_name2", type=str)
    parser.add_argument("--n_batches", type=int)
    parser.add_argument("--batch_size", type=int)

    parser.add_argument("--gram_or_cov", type=str)
    args = parser.parse_args()
    model_names = [args.model_name1, args.model_name2]
    task_list = get_perms(model_names)
    # imagenet_val_path = f"{paths['data_path']}/imagenet/val"
    transform = get_usual_transform()
    loader = setup_full_dataloader(args.batch_size, paths)
    model_cls1 = getattr(models, args.model_name1)
    model1 = model_cls1(pretrained=True).to(device).eval()
    model_cls2 = getattr(models, args.model_name2)
    model2 = model_cls2(pretrained=True).to(device).eval()
    # master_workers_queue(task_list, CCA_core, *(model_names, args.pooling, args.num_components, paths))
    #    a = batch_cka_core(1, layer_names, model_names, model1, model2, loader, n_batches, "gram", device, paths)
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
