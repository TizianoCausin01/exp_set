#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1 # number of processes
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --account=Sis25_piasini       # account name
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --job-name=CCA_alexnet
#SBATCH --output=/leonardo/home/userexternal/tcausin0/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id*
cd /leonardo/home/userexternal/tcausin0/exp_set/python_scripts/scripts/leonardo_scripts
module load python/3.11.6--gcc--8.5.0
source ~/virtual_envs/ponce_env/bin/activate
python3 cca_loop_within.py --model_name alexnet --pooling maxpool --num_components 50
# python feat_extraction.py --model_name $1 --num_images $2 --batch_size $3 --num_workers $SLURM_NTASKS --pooling $4
