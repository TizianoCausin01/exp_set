# from collections import defaultdict
import sys
sys.path.append("/leonardo/home/userexternal/tcausin0/exp_set/python_scripts/src")
from dim_redu_anns.incremental_pca import run_ipca_maxpool
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Incremental PCA for CNN layers")
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_components', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    run_ipca_maxpool(
        model_name=args.model_name,
        n_components=args.n_components,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
