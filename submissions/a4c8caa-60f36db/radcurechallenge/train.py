import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from .model.deepmtlr import DeepMTLR
from .utils import LogBestMetricsToCSV, LogEarlyStopMetricsToCSV


RANDOM_SEED = 42
seed_everything(RANDOM_SEED)


def main(model, hparams):
    version = os.environ.get("SLURM_ARRAY_JOB_ID")
    if version is None:
        version = os.environ.get("SLURM_JOB_ID")
    # support array jobs, e.g. for hyperparameter search
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if version and slurm_task_id:
        version += "_" + slurm_task_id

    logger = TensorBoardLogger(hparams.logdir,
                               name=hparams.exp_name,
                               version=version)
    if hparams.num_checkpoints > 0:
        checkpoint_path = os.path.join(logger.experiment.get_logdir(),
                                       "checkpoints",
                                       "{epoch:02d}-{val_loss:.2e}-{roc_auc:.2f}")
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                              save_top_k=hparams.num_checkpoints,
                                              monitor="roc_auc",
                                              mode="max")
    else:
        checkpoint_callback = None

    if hparams.early_stop_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=hparams.early_stop_delta,
            patience=hparams.early_stop_patience,
            verbose=True,
        )
        log_best_metrics_callback = LogEarlyStopMetricsToCSV(logger.experiment.get_logdir(),
                                                             hparams.early_stop_patience)
    else:
        early_stop_callback = None
        log_best_metrics_callback = LogBestMetricsToCSV("validation/surv/roc_auc_at_2yrs",
                                                        logger.experiment.get_logdir())
    callbacks = []
    callbacks.append(log_best_metrics_callback)
    model = model(hparams)
    trainer = Trainer.from_argparse_args(hparams,
                                         deterministic=True,
                                         benchmark=False,
                                         logger=logger,
                                         checkpoint_callback=checkpoint_callback,
                                         early_stop_callback=early_stop_callback,
                                         callbacks=callbacks)
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    model = DeepMTLR
    parser = ArgumentParser()
    parser = model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("root_directory",
                        type=str,
                        help="Directory containing images and segmentation masks.")
    parser.add_argument("emr_data_path",
                        type=str,
                        help="Path to CSV file containing the EMR data.")
    parser.add_argument("--logdir",
                        type=str,
                        default="./data/logs",
                        help="Directory where training logs will be saved.")
    parser.add_argument("--num_workers",
                        type=int,
                        default=1,
                        help="Number of worker processes to use for data loading.")
    parser.add_argument("--exp_name",
                        type=str,
                        default="deepmtlr",
                        help="Experiment name for logging purposes.")
    parser.add_argument("--early_stop_delta",
                        type=float,
                        default=.01)
    parser.add_argument("--early_stop_patience",
                        type=int,
                        default=10)
    parser.add_argument("--num_checkpoints",
                        type=int,
                        default=5)
    parser.add_argument("--pred_save_path",
                        type=str,
                        default="predictions.csv")


    hparams = parser.parse_args()
    main(model, hparams)
