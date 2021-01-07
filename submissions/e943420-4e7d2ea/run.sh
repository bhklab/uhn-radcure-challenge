#!/bin/bash

. ./shell_scripts/train_cnn.sh
. ./shell_scripts/train_dual.sh
. ./shell_scripts/predict_cnn.sh
. ./shell_scripts/predict_dual.sh

# added by organizers
RADIOMICS_SUBMISSION_ID="c22638c"
COMBINED_SUBMISSION_ID="5e515f3"
cp data/predictions/xl_1079790.csv ../../predictions/challenge_${RADIOMICS_SUBMISSION_ID}.csv
cp data/predictions/dual.csv ../../predictions/challenge_${COMBINED_SUBMISSION_ID}.csv
