# RADCURE Challenge submission
Survival prediction in head and neck cancer using multi-task logistic regression.

## Setup
The recommended way is to create a Conda virtual environment using the provided requirements file:
```bash
conda env create -f environment.yml --prefix ./env
conda activate ./env
```
Alternatively, install all of the packages listed in environment.yml manually.

## Training and prediction
The provided script (`run.sh`) can be used to train the EMR-only and combined models and save the predictions.
