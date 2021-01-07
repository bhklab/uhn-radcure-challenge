# UHN Survival Prediction Challenge Participant Code

## Getting Started
### Using Conda:
1. [Install conda](https://docs.conda.io/en/latest/miniconda.html)
2. Create the virtual environment:
'''CFLAGS='-std=c++11' conda env create -f environment.yml'''
3. Activate the environment:
'''conda activate radcure-challenge'''

### Extract Hand-Engineered Radiomic Features:
1. [Pyradiomics](https://pyradiomics.readthedocs.io/en/latest/)

## Using The Code:
### Update Data Paths:
The training and evaluation contained require seperate csv files for EMR and radiomic features. 
Within 'radcurechallenge/run_logistic_fuzzy_open.py' please update the following. 
1. Change line 12 to point to your radiomic features.
2. Change line 13 to point to your EMR file.
3. Change line 14 to point to your desired output location. 

### To-Run:
sbatch 'radcurechallenge/RUNME.sh'

OR

python 'run_logistic_fuzzy_open.py'
