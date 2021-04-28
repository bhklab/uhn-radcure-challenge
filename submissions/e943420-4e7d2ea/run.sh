#!/bin/bash

. ./shell_scripts/train_cnn.sh
. ./shell_scripts/train_dual.sh
. ./shell_scripts/predict_cnn.sh
. ./shell_scripts/predict_dual.sh

# added by organizers
RADIOMICS_SUBMISSION_ID="4e7d2ea"
COMBINED_SUBMISSION_ID="e943420"
cp data/predictions/xl_1079790.csv ../../predictions/challenge_radiomics_${RADIOMICS_SUBMISSION_ID}.csv
cp data/predictions/dual.csv ../../predictions/challenge_combined_${COMBINED_SUBMISSION_ID}.csv
