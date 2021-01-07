#!/bin/bash

python search.py\
    "python -m radcurechallenge.train\
          data/images\
          ../../data/emr.csv"\
    --config_file ../data/hyperparams/hyperparams_combined.yaml\
    --hparams_save_path ../data/hyperparams/\
    --num_samples 60\
    --max_concurrent_jobs 4\
    --source_bashrc\
    --conda_env ../env\
