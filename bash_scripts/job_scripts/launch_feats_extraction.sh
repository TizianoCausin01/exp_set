#!/bin/bash
# $1 -> the models to test, in this format "model1 model2 ..."
# $2 -> the number of images to compute, select 0 if you want them all
# $3 -> the batch_size
# $4 -> pooling : all , avgpool , maxpool
# $5 ->mobilenet_opt
num_images=$2
batch_size=$3
pooling=$4
mobilenet_opt=$5
read -a models <<< "$1"    # reads the first argin and creates an array called files
for m in "${models[@]}"; do
    sbatch --job-name=${m}_features_extraction ./general_feats_extraction.sh $m $2 $3 $4 $5
done
