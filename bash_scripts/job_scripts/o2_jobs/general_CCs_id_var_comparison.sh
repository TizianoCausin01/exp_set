#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --ntasks=1 # number of processes
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
# #SBATCH --account=       # account name
#SBATCH --partition=short # partition name
#SBATCH --job-name=CCs_var_id_comparison
#SBATCH --output=/home/tic569/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id*
cd /home/tic569/exp_set/python_scripts/scripts
module load conda/miniforge3/24.11.3-0
module load gcc/14.2.0
module load python/3.13.1
module load openmpi/4.1.8
conda activate ponce_env

python3 run_CCs_var_id_estimate.py --model_name1 $1 --model_name2 $2 --layer_name1 $3 --layer_name2 $4 --n_samples $5 --test_model $6 --test_layer $7 --batch_size $8 --neurons_perc $9


