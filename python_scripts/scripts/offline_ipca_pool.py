# to run like this e.g. 
# python offline_ipca_pool.py --model_name alexnet --pooling avgpool --n_components 1000 --batch_size 102

import os, yaml, sys
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from datetime import datetime
from dim_redu_anns.incremental_pca import offline_ipca_pool
import argparse


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--model_name', type=str, default='resnet18')
        parser.add_argument('--pooling', type=str, default="all")
        parser.add_argument('--n_components', type=int, default=1000)
        parser.add_argument('--batch_size', type=int, default=512)
        args = parser.parse_args()
        offline_ipca_pool(model_name=args.model_name, pooling=args.pooling, n_components=args.n_components, batch_size=args.batch_size, results_path=paths["results_path"])

