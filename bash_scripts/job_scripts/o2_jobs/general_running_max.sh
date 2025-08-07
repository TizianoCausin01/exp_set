#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1 # number of processes
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
# #SBATCH --account=       # account name
#SBATCH --partition=short # partition name
#SBATCH --job-name=running_max
#SBATCH --output=/home/tic569/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id*
cd /home/tic569/exp_set/python_scripts/scripts
module load conda/miniforge3/24.11.3-0
module load gcc/14.2.0
module load python/3.13.1
conda activate ponce_env
python3 run_running_max.py --model_name $1 --extreme_n_imgs $2 --top_n_PCs $3 --num_stim $4 --batch_size $5 --num_workers $SLURM_NTASKS 
