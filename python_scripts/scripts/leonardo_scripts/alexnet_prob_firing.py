import torch
import sys
sys.path.append("/leonardo/home/userexternal/tcausin0/exp_set/python_scripts/src")
from sparsity_in_silico.sparsity_CNN import response_prob_np
import h5py


path2res = "/leonardo_work/Sis25_piasini/tcausin/exp_set_res/silico"
with h5py.File(f"{path2res}/feats_alexnet.h5", 'r') as f:
    p = response_prob_np(f)

with h5py.File(f"{path2res}/prob_alexnet.h5", "w") as f:
# Iterate over dictionary items and save them in the HDF5 file
    for key, value in p.items():
        f.create_dataset(
            key, data=value
        )  # Create a dataset for each key-value pair
