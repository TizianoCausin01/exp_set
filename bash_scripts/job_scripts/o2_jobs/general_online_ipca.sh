#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --time=24:00:00 
#SBATCH --ntasks=4 # number of processes 
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=1 
#SBATCH --mem=250G 
# #SBATCH --account=       # account name 
# #SBATCH --partition=priority # partition name 
#SBATCH --job-name=online_ipca 
#SBATCH --output=/home/tic569/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id* 
cd /home/tic569/exp_set/python_scripts/scripts 
module load conda/miniforge3/24.11.3-0 
module load gcc/14.2.0 
module load python/3.13.1 
conda activate ponce_env 
python3 run_online_iPCA.py --model_name $1 --n_components $2 --batch_size $3 --num_workers $SLURM_NTASKS
