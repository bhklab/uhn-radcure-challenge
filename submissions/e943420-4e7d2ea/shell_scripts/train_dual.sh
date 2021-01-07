#!/bin/bash

USERNAME=""
RAW_DATA_PATH=""

conda activate env

python -m surv_challenge.dual.train\
          $RAW_DATA_PATH\
          $RAW_DATA_PATH/training/clinical.csv\
          --cache_dir ./data/data_cache\
          --num_workers 16\
          --batch_size 10\
          --lr 1e-3\
          --weight_decay 1e-4\
          --max_epochs 500\
          --gpus 1\
          --exp_name phase2\
          --design dropzero\
