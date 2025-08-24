import os, yaml, sys
import joblib
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
    parser.add_argument('--model_name1', type=str)
    parser.add_argument('--layer_name1', type=str)
    parser.add_argument('--model_name2', type=str)
    parser.add_argument('--layer_name2', type=str)
    parser.add_argument('--n_levels', type=int)
    parser.add_argument('--test_model', type=str)
    parser.add_argument('--test_layer', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--neurons_perc', type=int) 
    args = parser.parse_args()
    model_name1 = "alexnet"; model_name2 = "vit_b_16"; 
    target_layer1 = "features.0"; target_layer2 = "encoder.layers.encoder_layer_2.add_1"
    pooling="PC_pool"

    d1, d2 = project_onto_CCs(args.model_name1, args.model_name2, args.layer_name1, args.layer_name2, pooling, 100, paths)
    n_samples = d1.shape[0]
    max_n_clusters = n_samples//1000
    n_clusters_per_level = [max_n_clusters // (2 ** i) for i in range(args.n_levels)]
    print(n_clusters_per_level)

    ID_var_estimate(args.model_name1, args.layer_name1, d1, n_clusters_per_level, args.test_model, args.test_layer, args.neurons_perc, args.batch_size, paths, alignment=True, model_name2=args.model_name2, layer_name2=args.layer_name2)
