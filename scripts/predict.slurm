#!/bin/bash
#SBATCH --job-name=simple_cnn
#SBATCH --mem=60G
#SBATCH -t 2:00:00
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -p gpu_radiomics
#SBATCH -A radiomics_gpu
#SBATCH --gres=gpu:1
#SBATCH -D /cluster/home/username/path/to/challenge/directory

USERNAME=$(whoami)
source /cluster/home/$USERNAME/.bashrc
# OPTIONAL: activate the conda environment
conda activate radcure-challenge

python -m radcurechallenge.baselines.cnn.predict path/to/saved/checkpoint.pt
