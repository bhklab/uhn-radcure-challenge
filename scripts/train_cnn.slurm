#!/bin/bash
### Example script for submitting GPU jobs on H4H
#SBATCH --job-name=simple_cnn
#SBATCH --mem=60G
#SBATCH -t 1-00:00:00
#SBATCH -c 16
#SBATCH -N 1
# add the name of the partition (e.g. -p gpu) and your group's GPU account
# (e.g. -A mygroup_gpu) below
#SBATCH -p gpu
#SBATCH -A mygroup_gpu
#SBATCH --gres=gpu:1
# replace the path below with the directory with your code
#SBATCH -D /cluster/home/username/path/to/challenge/directory

USERNAME=$(whoami)
source /cluster/home/$USERNAME/.bashrc
# OPTIONAL: activate the conda environment
conda activate radcure-challenge

RAW_DATA_PATH=/cluster/projects/mygroup/RADCURE-challenge/data

python -m radcurechallenge.baselines.cnn.train\
          $RAW_DATA_PATH\
          $RAW_DATA_PATH/training/clinical.csv\
          --cache_dir ./data/data_cache\
          --num_workers 16\
          --batch_size 16\
          --lr 1e-3\
          --weight_decay 1e-4\
          --max_epochs 500\
          --gpus 1\
