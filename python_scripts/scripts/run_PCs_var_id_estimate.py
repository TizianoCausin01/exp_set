import os, yaml, sys
import joblib
import numpy as np
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from sampling.sampling_comparisons import ID_var_estimate
from experiments.utils import project_onto_CCs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--layer_name', type=str)
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--test_model', type=str)
    parser.add_argument('--test_layer', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--neurons_perc', type=int) 
    args = parser.parse_args()
    
    pooling="PC_pool"
    d1 = joblib.load(f"{paths['results_path']}/imagenet_val_{args.model_name}_{args.layer_name}_{pooling}_features.pkl") 
    n_clusters_per_level = [5000]
    mid_level = (np.log10(n_clusters_per_level[0]) + np.log10(args.n_samples))/2
    n_clusters_per_level.extend([int(np.round(10**mid_level)), args.n_samples])
    print(n_clusters_per_level)

    paths = paths

    ID_var_estimate(args.model_name, args.layer_name, d1, n_clusters_per_level, args.test_model, args.test_layer, args.neurons_perc, args.batch_size, paths, alignment=False, model_name2=None, layer_name2=None)
