import os, sys, time
import numpy as np
import joblib
import torch
from sklearn.cross_decomposition import CCA
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from datetime import datetime
sys.path.append("/leonardo/home/userexternal/tcausin0/exp_set/python_scripts/src")
from dim_redu_anns.utils import get_relevant_output_layers
from alignment.CCA import CCA_loop_within_modd
import argparse

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--model_name', type=str, default='resnet18')
        parser.add_argument('--pooling', type=str, default="all")
        parser.add_argument('--num_components', type=int, default=50)
        args = parser.parse_args()
        res_path = "/leonardo_work/Sis25_piasini/tcausin/exp_set_res/silico"
        CCA_loop_within_modd(args.model_name, args.pooling, args.num_components,True, res_path)
