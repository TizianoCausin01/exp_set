#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1 # number of processes
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --account=Sis25_piasini       # account name
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --job-name=CCA_between
#SBATCH --output=/leonardo/home/userexternal/tcausin0/output_jobs/%x.%j.out   # file name will be *job_name*.*job_id*
cd /leonardo/home/userexternal/tcausin0/exp_set/python_scripts/scripts
module load python
source ~/virtual_envs/ponce_env/bin/activate
python3 cca_between_mod.py --model_name1 $1 --model_name2 $2 --pooling $3 --n_components $4
