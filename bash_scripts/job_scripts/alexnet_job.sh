#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1 # number of processes
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --account=Sis25_piasini       # account name
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --job-name=exp_set_alexnet_feats
#SBATCH --output=/leonardo/home/userexternal/tcausin0/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id*
cd /leonardo/home/userexternal/tcausin0/exp_set/python_scripts/scripts/leonardo_scripts
module load python/3.11.6--gcc--8.5.0
source ~/virtual_envs/ponce_env/bin/activate
python /leonardo/home/userexternal/tcausin0/exp_set/python_scripts/scripts/leonardo_scripts/alexnet_feat_extraction.py 
