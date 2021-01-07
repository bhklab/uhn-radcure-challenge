#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate radcure-challenge

# EMR-only submission
python -m radcurechallenge.train

# combined-volume submission
python -m radcurechallenge.train --combined

# added by organizers
EMR_SUBMISSION_ID="c22638c"
COMBINED_SUBMISSION_ID="5e515f3"
cp predictions_emr.csv ../../predictions/challenge_${EMR_SUBMISSION_ID}.csv
cp predictions_combined.csv ../../predictions/challenge_${COMBINED_SUBMISSION_ID}.csv
