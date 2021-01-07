## Description
Training code for the RADCURE 2020 challenge submission.

The `run.sh` script can be used to prepare the data and train image-only and combined models with the same hyperparameters as the challenge submissions. Tensorboard logs will be created under `data/logs`. Additionally, the [scripts](scripts) directory contains scripts used for hyperparameter search on a [Slurm cluster](https://slurm.schedmd.com/documentation.html). Training requires a modern NVIDIA GPU (tested on Tesla P100), while the hyperparameter search requires a working Slurm setup and ideally more than 1 GPU (tested on 4x Tesla P100, takes about 4 GPU-days).


## How to run
### Using the training script (recommended)
Simply execute the training script, it will create a  virtual environment and install the required packages:
 ```bash
./run.sh 
 ```
 
### Manually
1. Install the required packages.
Using pip:
```bash
pip install -r requirements.txt
```
Using conda:
```bash
conda install --file requirements.txt
```
2. Prepare the images:
```bash
python radcurechallenge/preprocess.py path/to/training/images data/images/training --show_progress
python radcurechallenge/preprocess.py path/to/test/images data/images/test --show_progress
```
3. Start the training:
```bash
python -m radcurechallenge.train data/images data/emr.csv --gpus 1 <other_hyperparams>
```
See [radcurechallenge/model/deepmtlr.py](radcurechallenge/model/deepmtlr.py) for all available hyperparameters.
