import joblib
import numpy as np
import pandas as pd
import joblib
import sys
sys.path.append("/leonardo/home/userexternal/tcausin0/exp_set/python_scripts/src")
from dim_redu_anns.utils import get_relevant_output_layers
from datetime import datetime

path2res = "/leonardo_work/Sis25_piasini/tcausin/exp_set_res/silico"
maxpool = 0
if maxpool==0:
    models = ["alexnet", "resnet18", "vit_b_16", "resnet50"] #substitute with manual input
else:
     models = ["alexnet", "resnet18", "resnet50"] # didn't do maxpool over vit

for m in models:
    layers = get_relevant_output_layers(m)
    for l in layers:
        if maxpool==1:
            path2file = f"{path2res}/imagenet_val_{m}_{l}_max_pool_pca_model_1000_PCs.pkl" 
        else:
            path2file = f"{path2res}/imagenet_val_{m}_{l}_pca_model_1000_PCs.pkl"     
        print(datetime.now().strftime("%H:%M:%S"), f"processing layer {l} of model {m}")
        data = joblib.load(path2file)
        df = pd.DataFrame(data_l0.explained_variance_ratio_)
        df.to_csv(f"{path2res}/var_explained_pca_{m}_{l}_1000_PCs.csv", index=False) 

