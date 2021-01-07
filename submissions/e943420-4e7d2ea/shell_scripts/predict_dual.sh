#!/bin/bash

USERNAME=""
TEST_DATA_PATH=""
CHECKPOINT_PATH=""

conda activate env

python -m radcurechallenge.dual.predict\
        $CHECKPOINT_PATH\
        ./data/predictions/dual.csv\
        $TEST_DATA_PATH\
        $TEST_DATA_PATH/clinical.csv\
        
