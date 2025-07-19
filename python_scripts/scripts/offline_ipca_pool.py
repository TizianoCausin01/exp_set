import os, yaml, sys
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
from datetime import datetime
from dim_redu_anns.incremental_pca import offline_ipca_pool

offline_ipca_pool(model_name="alexnet", pooling="maxpool", n_components=1000, batch_size=512, results_path=paths["results_path"])

