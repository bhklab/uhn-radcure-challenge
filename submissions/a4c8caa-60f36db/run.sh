# install requirements
conda create --prefix ./env -f environment.yml
conda activate ./env

# preprocess the images
python radcurechallenge/preprocess.py\
       ../../data/images/training\
       ./data/images/training\
       --show_progress

python radcurechallenge/preprocess.py\
       ../../data/images/test\
       ./data/images/test\
       --show_progress

# train the image-only model
python -m radcurechallenge.train\
       data/images\
       ../../data/emr.csv\
       --num_workers 32\
       --batch_size 16\
       --lr 2.7e-4\
       --weight_decay 1.7e-4\
       --dropout .03\
       --max_steps 1470\
       --gpus 1\
       --growth_rate 24\
       --lr_decay_factor .5\
       --lr_decay_milestones 60 100 140 180\
       --balance_batches\
       --num_time_bins 40\
       --mtlr_smooth_factor 10\
       --two_stream\
       --num_checkpoints 1\
       --early_stop_patience 0\
       --exp_name cnn_overall\
       --pred_save_path radiomics.csv

# train the combined model
python -m radcurechallenge.train\
       data/images\
       ../../data/emr.csv\
       --num_workers 32\
       --batch_size 8\
       --lr 1.95e-4\
       --weight_decay 9.4e-4\
       --dropout .38\
       --max_steps 11500\
       --gpus 1\
       --growth_rate 24\
       --lr_decay_factor .5\
       --lr_decay_milestones 60 100 140 180\
       --balance_batches\
       --use_emr_info\
       --num_time_bins 40\
       --mtlr_smooth_factor 10\
       --num_checkpoints 1\
       --early_stop_patience 0\
       --exp_name cnn_overall_clinical\
       --pred_save_path combined.csv

# added by organizers
RADIOMICS_SUBMISSION_ID="60f36db"
COMBINED_SUBMISSION_ID="a4c8caa"
cp radiomics.csv ../../predictions/challenge_radiomics_${RADIOMICS_SUBMISSION_ID}.csv
cp combined.csv ../../predictions/challenge_combined_${COMBINED_SUBMISSION_ID}.csv
