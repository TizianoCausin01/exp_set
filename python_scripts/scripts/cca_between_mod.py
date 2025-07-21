# to run like this e.g. 
# python offline_ipca_pool.py --model_name alexnet --pooling avgpool --n_components 1000 --batch_size 102

import os, yaml, sys
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from alignment.CCA import CCA_loop_between_mod
from datetime import datetime
import argparse



if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--model_name1', type=str, default='resnet18')
        parser.add_argument('--model_name2', type=str, default='resnet18')
        parser.add_argument('--pooling', type=str, default="all")
        parser.add_argument('--n_components', type=int, default=50)
        parser.add_argument('--pca_option', type=bool, default=True)
        args = parser.parse_args()
        CCA_loop_between_mod([args.model_name1, args.model_name2], args.pooling, args.n_components, args.pca_option, paths["results_path"])
