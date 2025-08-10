from mpi4py import MPI
import os, yaml, sys

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from parallel.parallel_funcs import master_workers_queue, print_wise
from dim_redu_anns.utils import get_relevant_output_layers
import numpy as np
import argparse

# from datetime import datetime


# import random
def function_to_do(task_list, idx):
    print_wise(task_list[idx])
    

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Incremental PCA for CNN layers")
    parser.add_argument("--model_name1", type=str)
    parser.add_argument("--model_name2", type=str)
    args = parser.parse_args()
    task_list = get_perms([args.model_name1, args.model_name2])
    master_workers_queue(
        task_list, function_to_do
    )
