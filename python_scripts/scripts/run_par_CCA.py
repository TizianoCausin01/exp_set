from mpi4py import MPI
import os, yaml, sys

ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from parallel.parallel_funcs import master_workers_queue, print_wise, get_perms, CCA_core
from dim_redu_anns.utils import get_relevant_output_layers
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Incremental PCA for CNN layers")
    parser.add_argument("--model_name1", type=str)
    parser.add_argument("--model_name2", type=str)
    parser.add_argument("--pooling", type=str)
    parser.add_argument("--num_components", type=int)
    args = parser.parse_args()
    model_names = [args.model_name1, args.model_name2]
    task_list = get_perms(model_names)
    master_workers_queue(task_list, CCA_core, *(model_names, args.pooling, args.num_components, paths)) 
    
