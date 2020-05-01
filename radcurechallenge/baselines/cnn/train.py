import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .model import SimpleCNN


def main(args):
    # XXX uncomment before release
    # if args.gpus == 0:
    #     raise ValueError(("Training on CPU is not supported, please run again "
    #                       "on a system with GPU and '--gpus' > 0."))

    checkpoint_path = os.path.join(args.checkpoint_dir,
                                   "simplecnn_{epoch:02d}-{tune_loss:.2f}-{roc_auc:.2f}")
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          save_top_k=5,
                                          monitor="tuning_loss")
    model = SimpleCNN(args, args.clinical_data_path, args.cache_dir, args.num_workers)
    trainer = Trainer(max_epochs=args.max_epochs,
                      weights_save_path=args.weights_save_path,
                      gpus=args.gpus,
                      fast_dev_run=args.fast_dev_run, # XXX remove before release
                      checkpoint_callback=checkpoint_callback)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--root_directory",
                        type=str,
                        default="./.model_checkpoints",
                        help="Directory containing images and segmentation masks.")
    parser.add_argument("--clinical_data_path",
                        type=str,
                        default="./.model_checkpoints",
                        help="Path to CSV file containing the clinical data.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="The batch size.")
    parser.add_argument("--lr",
                        type=float,
                        default=3e-4,
                        help="The initial learning rate.")
    parser.add_argument("--patch_size",
                        type=int,
                        default=50,
                        help="Size of the image patch extracted around each tumour.")
    parser.add_argument("--checkpoint_dir",
                        type=str,
                        default="./.model_checkpoints",
                        help="Directory where model checkpoints will be saved.")
    parser.add_argument("--cache_dir",
                        type=lambda x: None if x == "None" else x,
                        nargs="?",
                        default=None,
                        help=("Directory where the preprocessed data "
                              "will be saved. If not specified, data will not "
                              "be cached, which slows down loading but conserves "
                              "disk space."))
    parser.add_argument("--num_workers",
                        type=int,
                        default=1,
                        help="Number of worker processes to use for data loading.")

    args = parser.parse_args()
    main(args)
