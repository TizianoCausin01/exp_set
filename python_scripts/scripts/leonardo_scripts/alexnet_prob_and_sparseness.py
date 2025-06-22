import sys
sys.path.append("/leonardo/home/userexternal/tcausin0/exp_set/python_scripts/src")
from sparsity_in_silico.sparsity_CNN import response_prob_np, rust_sparseness
from sparsity_in_silico.utils import get_data_mat
import h5py


path2res = "/leonardo_work/Sis25_piasini/tcausin/exp_set_res/silico"
with h5py.File(f"{path2res}/feats_alexnet.h5", 'r') as f:
    p = {}
    S = {}
    for k in f.keys(): 
        data_mat = get_data_mat(f, k)
        p[k] = response_prob(data_mat)
        S[k] = rust_sparseness(data_mat)

with h5py.File(f"{path2res}/prob_alexnet.h5", "w") as f:
# Iterate over dictionary items and save them in the HDF5 file
    for key, value in p.items():
        f.create_dataset(
            key, data=value
        )  # Create a dataset for each key-value pair


with h5py.File(f"{path2res}/rust_sparseness_alexnet.h5", "w") as f:
# Iterate over dictionary items and save them in the HDF5 file
    for key, value in S.items():
        f.create_dataset(
            key, data=value
        )  # Create a dataset for each key-value pair
