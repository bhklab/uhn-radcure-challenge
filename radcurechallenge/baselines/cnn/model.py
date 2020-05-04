from math import floor
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from .dataset import RadcureDataset
from .transforms import *


class SimpleCNN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool1 = nn.MaxPool3d(kernel_size=3)
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv4 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3)
        self.pool2 = nn.MaxPool3d(kernel_size=3)

        self.global_pool = nn.AdaptiveAvgPool3d(512)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fc = nn.Linear(512, 1)

        self.conv_bn1 = nn.BatchNorm3d(self.conv1.out_channels)
        self.conv_bn2 = nn.BatchNorm3d(self.conv2.out_channels)
        self.conv_bn3 = nn.BatchNorm3d(self.conv3.out_channels)
        self.conv_bn4 = nn.BatchNorm3d(self.conv4.out_channels)

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.leaky_relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv_bn3(x)
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = self.conv_bn4(x)
        x = self.leaky_relu(x)
        x = self.pool2(x)

        x = self.global_pool(x)

        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.xavier_uniform_(m.bias)

    def prepare_data(self):
        train_dataset = RadcureDataset(self.hparams.root_directory,
                                       self.hparams.clinical_data_path,
                                       self.hparams.patch_size,
                                       train=True,
                                       cache_dir=self.hparams.cache_dir)
        valid_dataset = RadcureDataset(self.hparams.root_directory,
                                      self.hparams.clinical_data_path,
                                      self.hparams.patch_size,
                                      train=False,
                                      cache_dir=None)
        # if self.hparams.cache_dir:
            # # load data into cache before training
            # dl = DataLoader(dataset,
            #                 batch_size=self.hparams.batch_size,
            #                 num_workers=self.hparams.num_workers)
            # # iterate over dataloader to load data into cache
            # for _ in dl:
            #     continue

        # make sure the tuning set is balanced
        tune_size = floor(.1 / .7 * len(train_dataset)) # use 10% of all data for tuning
        train_indices = range(len(train_dataset))
        train_targets = train_dataset.clinical_data["target_binary"]
        train_indices, tune_indices = train_test_split(train_indices, test_size=tune_size, stratify=train_targets)
        train_dataset, tune_dataset = Subset(train_dataset, train_indices), Subset(train_dataset, tune_indices)

        # apply data augmentation only on training set
        train_transform = Compose([
            ToTensor(),
            Normalize(self.hparams.dataset_mean, self.hparams.dataset_std),
            Random90DegRotation(),
            RandomFlip((0, 1)),
            RandomFlip((0, 2)),
            RandomFlip((1, 2)),
            RandomNoise()
        ])
        valid_transform = Compose([
            ToTensor(),
            Normalize(self.hparams.dataset_mean, self.hparams.dataset_std)
        ])
        train_dataset.dataset.transform = train_transform
        tune_dataset.dataset.transform = valid_transform
        valid_dataset.transform = valid_transform
        self.train_dataset = train_dataset
        self.tune_dataset = tune_dataset
        self.valid_dataset = valid_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.tune_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.hparams.lr,
                         weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(output,
                                                  y,
                                                  pos_weight=self.hparams.pos_weight)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(output,
                                                  y,
                                                  pos_weight=self.hparams.pos_weight)
        pred_prob = F.sigmoid(output)
        return {"loss": loss, "pred_prob": pred_prob, "y": y}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        pred_prob = torch.stack([x["pred_prob"] for x in outputs]).detach().cpu().numpy()
        y = torch.stack([x["y"] for x in outputs]).detach().cpu().numpy()
        roc_auc = roc_auc_score(y, pred_prob)
        avg_prec = average_precision_score(y, pred_prob)
        # log loss and metrics to Tensorboard
        log = {"tuning_loss": loss, "roc_auc": roc_auc, "average_precision": avg_prec}
        return {"tuning_loss": loss, "log": log}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size",
                            type=int,
                            default=16,
                            help="The batch size.")
        parser.add_argument("--lr",
                            type=float,
                            default=3e-4,
                            help="The initial learning rate.")
        parser.add_argument("--weight_decay",
                            type=float,
                            default=1e-5,
                            help="The amount of weight decay to use.")
        parser.add_argument("--patch_size",
                            type=int,
                            default=50,
                            help=("Size of the image patch extracted around"
                                  "each tumour."))
        parser.add_argument("--dataset_mean",
                            type=float,
                            default=0., # XXX add correct value
                            help=("The mean pixel intensity used for"
                                  "input normalization."))
        parser.add_argument("--dataset_std",
                            type=float,
                            default=1., # XXX add correct value
                            help=("The standard deviation of  pixel intensity"
                                  "used for input normalization."))
        parser.add_argument("--pos_weight",
                            type=float,
                            default=1e-5, # XXX add corect value
                            help="The loss weight for positive class.")
        return parser
