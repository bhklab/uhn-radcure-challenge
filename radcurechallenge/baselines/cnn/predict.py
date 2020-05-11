from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer

from .model import SimpleCNN

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):
    # if hparams.gpus == 0:
    #     raise ValueError(("Training on CPU is not supported, please run again "
    #                       "on a system with GPU and '--gpus' > 0."))

    model = SimpleCNN.load_from_checkpoint(args.checkpoint_path)
    print(model)
    print(model.hparams)
    model.prepare_data()
    trainer = Trainer.from_argparse_args(model.hparams)
    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("checkpoint_path",
                        type=str,
                        help="Path to saved model checkpoint.")
    parser.add_argument("--pred_save_path",
                        type=str,
                        default="./data/predictions/baseline_cnn.csv",
                        help="Directory where final predictions will be saved.")

    args = parser.parse_args()
    main(args)
