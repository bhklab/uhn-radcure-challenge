#!/bin/bash

python -u ./run_logistic_fuzzy_open.py

# added by organizers
COMBINED_SUBMISSION_ID="c985b3d"
COMBINED_VOLUME_SUBMISSION_ID="c927742"
RADIOMICS_SUBMISSION_ID="5af6c56"
cp fuzzyVol_clin+rad_currated_test_V02.csv ../../predictions/challenge_${COMBINED_SUBMISSION_ID}.csv
cp fuzzyVol_clin_test_V02.csv ../../predictions/challenge_${COMBINED_VOLUME_SUBMISSION_ID}.csv
cp fuzzyVol_rad_currated_test_V02.csv ../../predictions/challenge_${RADIOMICS_SUBMISSION_ID}.csv
