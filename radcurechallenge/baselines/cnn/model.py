from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn.optim import Adam
from torchvision.transforms import Compose

from sklearn.metrics import roc_auc_score, average_precision_score

from .dataset import RadcureDataset
from .transforms import *


class SimpleCNN(pl.LightningModule):
    def __init__(self,
                 hparams,
                 root_directory,
                 clinical_data_path,
                 cache_dir="./.cache",
                 num_workers=1):
        super().__init__()

        self.hparams = hparams
        self.root_directory = root_directory
        self.clinical_data_path = clinical_data_path
        self.cache_dir = cache_dir
        self.num_workers = num_workers

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
        dataset = RadcureDataset(self.root_directory,
                                 self.clinical_data_path,
                                 self.hparams.patch_size,
                                 cache_dir=self.cache_dir)
        if self.cache_dir:
            # load data into cache before training
            dl = DataLoader(dataset,
                            batch_size=self.hparams.batch_size,
                            num_workers=self.num_workers)
            # iterate over dataloader to load data into cache
            for _ in dl:
                continue

        test_size = floor(.1 / .7 * len(dataset)) # use 10% of all data for tuning
        train_dataset, tune_dataset = random_split(dataset, [len(dataset) - test_size, test_size])
        transform = Compose([
            ToTensor(),
            Normalize(self.hparams.dataset_mean, self.hparams.dataset_std),
            Random90DegRotation(),
            RandomFlip((0, 1)),
            RandomFlip((0, 2)),
            RandomFlip((1, 2)),
            RandomNoise(.1)
        ])
        train_dataset.transform = transform # only augment the training set
        self.train_dataset = train_dataset
        self.tune_dataset = tune_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.tune_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def configure_optimizers(self):
        adam = Adam(self.parameters(),
                    lr=self.hparams.learning_rate,
                    weight_decay=self.hparams.weight_decay)
        return adam

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
