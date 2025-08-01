#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1 # number of processes
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=480G
#SBATCH --account=Sis25_piasini       # account name
#SBATCH --partition=boost_usr_prod # partition name
# #SBATCH --job-name=giordano_alexnet_PCA
#SBATCH --output=/leonardo/home/userexternal/tcausin0/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id*
cd /leonardo/home/userexternal/tcausin0/exp_set/python_scripts/scripts/leonardo_scripts
module load python
source ~/virtual_envs/ponce_env/bin/activate
python feat_extraction.py --model_name $1 --num_images $2 --batch_size $3 --num_workers $SLURM_NTASKS --pooling $4 --rand_perc $5 --mobilenet_opt $6
