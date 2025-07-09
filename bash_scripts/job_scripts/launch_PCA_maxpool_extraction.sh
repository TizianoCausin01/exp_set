#!/bin/bash
# $1 -> the models to test, in this format "model1 model2 ..."
# $2 -> the number of PCs to compute
PCs=$2
read -a models <<< "$1"    # reads the first argin and creates an array called files
for m in "${models[@]}"; do
    sbatch --job-name=${m}_PCA_${PCs}_PCs ./general_maxpool_PCA.sh $m $PCs
done
