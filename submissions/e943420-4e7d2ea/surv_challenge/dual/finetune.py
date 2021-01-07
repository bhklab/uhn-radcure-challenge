import os
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .model import Challenger

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(hparams):
    hparams.design = hparams.path.split ("/")[-3].split("_")[0]
    
    slurm_id = os.environ.get("SLURM_JOBID")
    if slurm_id is None:
        version = None
    else:
        version = str(hparams.design +"_" + slurm_id)
    logger = TensorBoardLogger(hparams.logdir,
                               name=hparams.exp_name,
                               version=version)
    checkpoint_path = os.path.join(logger.experiment.get_logdir(),
                                   "checkpoints",
                                   "dual_{epoch:02d}-{val_loss:.2e}-{roc_auc:.2f}")
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          save_top_k=5,
                                          monitor="roc_auc",
                                          mode="max")
    model = Challenger.load_from_checkpoint(checkpoint_path=hparams.path)
    
    
    trainer = Trainer.from_argparse_args(hparams)
    trainer.logger = logger
    trainer.checkpoint_callback = checkpoint_callback
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Challenger.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("root_directory",
                        type=str,
                        help="Directory containing images and segmentation masks.")
    parser.add_argument("clinical_data_path",
                        type=str,
                        help="Path to CSV file containing the clinical data.")
    parser.add_argument("--logdir",
                        type=str,
                        default="./data/logs",
                        help="Directory where training logs will be saved.")
    parser.add_argument("--cache_dir",
                        type=str,
                        default="./data/data_cache",
                        help=("Directory where the preprocessed data "
                              "will be saved."))
    parser.add_argument("--pred_save_path",
                        type=str,
                        default="./data/predictions/baseline_cnn.csv",
                        help="Directory where final predictions will be saved.")
    parser.add_argument("--num_workers",
                        type=int,
                        default=1,
                        help="Number of worker processes to use for data loading.")
    parser.add_argument("--exp_name",
                        type=str,
                        default="challenger",
                        help="Experiment name for logging purposes.")
    parser.add_argument ("--path",
                        type=str,
                        default="",
                        help="Model checkpoint absolute path")
                         

    hparams = parser.parse_args()
    main(hparams)
