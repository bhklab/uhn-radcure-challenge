#!/bin/bash

USERNAME=""
RAW_DATA_PATH=""
CHECKPOINT_PATH=""

source /cluster/home/$USERNAME/.bashrc
conda activate env

python -m radcurechallenge.dual.finetune\
          $RAW_DATA_PATH\
          $RAW_DATA_PATH/training/clinical.csv\
          --cache_dir ./data/data_cache\
          --num_workers 16\
          --batch_size 10\
          --lr 5e-5\
          --weight_decay 4e-5\
          --max_epochs 500\
          --gpus 1\
          --exp_name name\
          --path $CHECKPOINT_PATH\
