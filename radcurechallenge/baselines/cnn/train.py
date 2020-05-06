import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from .model import SimpleCNN

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(hparams):
    # XXX uncomment before release
    # if args.gpus == 0:
    #     raise ValueError(("Training on CPU is not supported, please run again "
    #                       "on a system with GPU and '--gpus' > 0."))

    checkpoint_path = os.path.join(hparams.logdir,
                                   "model_checkpoints",
                                   "simplecnn_{epoch:02d}-{tune_loss:.2e}-{roc_auc:.2f}")
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          save_top_k=5,
                                          monitor="tuning_loss")
    logger = TestTubeLogger(os.path.join(hparams.logdir, "test_tube_logs"), name='simplecnn')
    model = SimpleCNN(hparams)
    trainer = Trainer.from_argparse_args(hparams,
                                         logger=logger,
                                         checkpoint_callback=checkpoint_callback)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
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
    parser.add_argument("--num_workers",
                        type=int,
                        default=1,
                        help="Number of worker processes to use for data loading.")

    parser = SimpleCNN.add_model_specific_args(parser)

    hparams = parser.parse_args()
    main(hparams)
