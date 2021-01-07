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
    model = Challenger.load_from_checkpoint(args.checkpoint_path)
    model.hparams.logger = None
    model.hparams.checkpoint_callback = None
    model.prepare_data()
    trainer = Trainer.from_argparse_args(hparams)
    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("checkpoint_path",
                        type=str,
                        help="Path to saved model checkpoint.")
    parser.add_argument("root_directory",
                        type=str,
                        help="Directory containing images and segmentation masks.")
    parser.add_argument("clinical_data_path",
                        type=str,
                        help="Path to CSV file containing the clinical data.")
    parser.add_argument("--pred_save_path",
                        type=str,
                        default="./data/predictions/baseline_cnn.csv",
                        help="Directory where final predictions will be saved.")

    args = parser.parse_args()
    main(args)
