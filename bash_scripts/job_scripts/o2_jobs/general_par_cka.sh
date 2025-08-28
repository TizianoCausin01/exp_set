#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks=20 # number of processes
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
# #SBATCH --account=       # account name
#SBATCH --partition=priority # partition name
#SBATCH --job-name=batch_cka
#SBATCH --output=/home/tic569/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id*
cd /home/tic569/exp_set/python_scripts/scripts
module load conda/miniforge3/24.11.3-0
module load gcc/14.2.0
module load python/3.13.1
module load openmpi/4.1.8
conda activate ponce_env
mpiexec -n $SLURM_NTASKS python3 run_batch_cka.py --model_name1 $1 --model_name2 $2 --n_batches $3 --batch_size $4 --gram_or_cov $5
