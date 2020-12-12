#!/bin/bash
#SBATCH --job-name=radcure_features
#SBATCH --mem=60G
#SBATCH -t 20:00:00
#SBATCH -c 10
#SBATCH -N 1
#SBATCH -p himem
#SBATCH -D /cluster/home/username/path/to/challenge/repo/directory

USERNAME=$(whoami)
source /cluster/home/$USERNAME/.bashrc
# OPTIONAL: activate the conda environment
conda activate radcure-challenge

RAW_DATA_PATH=/cluster/projects/mygroup/RADCURE-challenge/data

python radcurechallenge/baselines/simple/radiomics/extract_features.py\
       $RAW_DATA_PATH\
       $RAW_DATA_PATH/training/clinical.csv\
       --output_path data/radiomics.csv\
       --params_file data/pyradiomics_params.yaml
