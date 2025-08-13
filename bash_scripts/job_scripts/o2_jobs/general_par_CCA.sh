#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=10 # number of processes
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
# #SBATCH --account=       # account name
#SBATCH --partition=priority # partition name
#SBATCH --job-name=parallel_CCA
#SBATCH --output=/home/tic569/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id*
cd /home/tic569/exp_set/python_scripts/scripts
module load conda/miniforge3/24.11.3-0
module load gcc/14.2.0
module load python/3.13.1
module load openmpi/4.1.8
conda activate ponce_env
mpiexec -n $SLURM_NTASKS python3 run_par_CCA.py --model_name1 $1 --model_name2 $2 --pooling $3 --num_components $4
