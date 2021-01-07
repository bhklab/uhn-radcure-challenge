#!/bin/bash

USERNAME=""
RAW_DATA_PATH=""
CHECKPOINT_PATH=""

conda activate env

python -m radcurechallenge.radiomics.test\
	$CHECKPOINT_PATH\
    $RAW_DATA_PATH\
	$RAW_DATA_PATH/training/clinical.csv\
    --pred_save_path ./data/predictions/xl_1079790.csv\
	--gpus 1\
