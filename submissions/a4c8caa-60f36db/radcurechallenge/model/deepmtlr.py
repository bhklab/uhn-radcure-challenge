from argparse import ArgumentParser, Namespace
from copy import copy
from math import floor

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from ..dataset import RadcureDataset
from ..transforms import *
from ..utils import integrated_brier_score, plot_predictions, make_time_bins, plot_weights
from .densenet import DenseNet3d, conv1x1x1
from .mtlr import *


class DeepMTLR(pl.LightningModule):
    def __init__(self, hparams: Namespace):
        """Initialize the module.

        Parameters
        ----------
        hparams
            `Namespace` object containing the model hyperparameters.
            Should usually be generated automatically by `argparse`.
        """
        super().__init__()

        self.hparams = hparams
        self.eval_times = np.linspace(0, 23, 24)

        self.stem1 = DenseNet3d(growth_rate=self.hparams.growth_rate,
                                dropout_p=self.hparams.dropout)

        if self.hparams.two_stream:
            self.stem2 = DenseNet3d(growth_rate=self.hparams.growth_rate,
                                    dropout_p=self.hparams.dropout)
            self.final_conv = conv1x1x1(self.stem1.out_channels * 2, self.stem1.out_channels)
        else:
            self.stem2 = None
            self.final_conv = None

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        surv_head_in = self.stem1.out_channels
        if self.hparams.use_emr_info:
            surv_head_in += self.hparams.num_emr_features

        self.surv_head = MTLRLayer(surv_head_in, self.hparams.num_time_bins + 1)
        self.surv_loss = MTLRLoss(self.hparams.num_time_bins + 1)
        self.apply(self.init_params)

    def forward(self, x):
        (x_hires, x_lores), emr = x
        out = self.stem1(x_hires)
        all_out = {}

        if self.hparams.two_stream:
            out_lores = self.stem2(x_lores)
            out = torch.cat([out, out_lores], dim=1)
            out = self.final_conv(out)

        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        if self.hparams.use_emr_info:
            out = torch.cat([out, emr], dim=1)
        all_out["surv"] = self.surv_head(out)
        return all_out

    def init_params(self, m: torch.nn.Module):
        """Initialize the parameters of a module.

        Parameters
        ----------
        m
            The module to initialize.

        Notes
        -----
        Convolutional layer weights are initialized from a normal distribution
        as described in [1]_ in `fan_in` mode. The final layer bias is
        initialized so that the expected predicted probability accounts for
        the class imbalance at initialization.

        References
        ----------
        .. [1] K. He et al. ‘Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification’,
           arXiv:1502.01852 [cs], Feb. 2015.
        """

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def prepare_data(self):
        """Preprocess the data and create training, validation and test
        datasets.

        This method is called automatically by pytorch-lightning.
        """
        hires_size = lores_size = self.hparams.patch_size
        hires_spacing = self.hparams.voxel_spacing
        lores_spacing = list(np.array(hires_spacing) / self.hparams.lores_scale)

        transform = Compose([
            SpatialTransform(hires_size, hires_spacing, lores_size, lores_spacing, augment=False),
            ToTensor()
        ])

        full_dataset = RadcureDataset(self.hparams.root_directory,
                                      self.hparams.emr_data_path,
                                      train=True,
                                      transform=transform)

        # make sure the validation set is balanced
        val_size = floor(.1 / .7 * len(full_dataset)) # use 10% of all data for validation
        full_indices = range(len(full_dataset))
        full_targets = full_dataset.targets["target_binary"]
        train_indices, val_indices = train_test_split(full_indices, test_size=val_size, stratify=full_targets)
        train_dataset, val_dataset = Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)

        train_times = full_dataset.targets.iloc[train_indices]["survival_time"]
        self.time_bins = make_time_bins(train_times, num_bins=self.hparams.num_time_bins)
        full_dataset.time_bins = self.time_bins

        # compute image statistics for normalization on the training set
        train_mean_hires = train_mean_lores = 0.
        train_std_hires = train_std_lores = 0.
        for b in DataLoader(train_dataset,
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers):
            train_mean_hires += b.image[0].mean(dim=(1, 2, 3, 4)).sum().item()
            train_mean_lores += b.image[1].mean(dim=(1, 2, 3, 4)).sum().item()
            train_std_hires += b.image[0].std(dim=(1, 2, 3, 4)).sum().item()
            train_std_lores += b.image[1].std(dim=(1, 2, 3, 4)).sum().item()

        train_mean_hires /= len(train_dataset)
        train_mean_lores /= len(train_dataset)
        train_std_hires /= len(train_dataset)
        train_std_lores /= len(train_dataset)

        train_mean_clin = full_dataset.emr_data.iloc[train_indices].values.mean(0)
        train_std_clin = full_dataset.emr_data.iloc[train_indices].values.std(0)

        emr_transform = ColumnNormalize(train_mean_clin, train_std_clin)

        val_transform = Compose([
            SpatialTransform(hires_size, hires_spacing, lores_size, lores_spacing, augment=False),
            Normalize([train_mean_hires, train_mean_lores, 0],
                      [train_std_hires, train_std_lores, 1]),
            ToTensor()
        ])

        # apply data augmentation only on training set
        if self.hparams.no_augmentation:
            train_transform = val_transform
        else:
            train_transform = Compose([
                RandomNoise((10, 0)),
                SpatialTransform(hires_size, hires_spacing,
                                 lores_size, lores_spacing),
                Normalize([train_mean_hires, train_mean_lores, 0],
                          [train_std_hires, train_std_lores, 1]),
                ToTensor()
            ])
        val_dataset.dataset = copy(full_dataset)
        val_dataset.dataset.transform = val_transform
        train_dataset.dataset.transform = train_transform
        train_dataset.dataset.emr_transform = emr_transform
        val_dataset.dataset.emr_transform = emr_transform

        if self.hparams.balance_batches:
            train_targets = [full_dataset.targets.iloc[i]["target_binary"] for i in train_indices]
            train_targets = torch.tensor(train_targets, dtype=torch.float)
            weights = torch.where(train_targets.bool(), 1 / train_targets.sum(), 1 / (train_targets == 0).sum().float())
            self.train_sampler = WeightedRandomSampler(weights, len(train_targets))
        else:
            self.train_sampler = None

        test_dataset = RadcureDataset(self.hparams.root_directory,
                                      self.hparams.emr_data_path,
                                      train=False,
                                      transform=val_transform,
                                      emr_transform=emr_transform,
                                      time_bins=self.time_bins)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def on_train_start(self):
        """This method is called automatically by pytorch-lightning."""
        print("Dataset sizes")
        print("=============")
        print(f"training:   {len(self.train_dataset)}")
        print(f"validation: {len(self.val_dataset)}")

        if self.logger is not None and self.hparams.plot:
            # plot a few example images from the training, validation
            # and test datasets
            datasets = {
                "training": self.train_dataset,
                "validation": self.val_dataset
            }
            for key, dataset in datasets.items():
                imgs_hires = []
                imgs_lores = []
                masks = []
                for i in range(5):
                    sample = dataset[i]
                    img_hires = sample.image[0]
                    img_hires = (img_hires - img_hires.min()) / (img_hires.max() - img_hires.min())
                    img_lores = sample.image[1]
                    img_lores = (img_lores - img_lores.min()) / (img_lores.max() - img_lores.min())
                    mask = sample.mask
                    imgs_hires.append(img_hires)
                    imgs_lores.append(img_lores)
                    masks.append(mask)
                imgs_hires = torch.stack(imgs_hires, dim=0)
                imgs_lores = torch.stack(imgs_lores, dim=0)
                masks = torch.stack(masks, dim=0)

                for i in range(imgs_hires.size(2)):
                    self.logger.experiment.add_images(key + "/hires",
                                                      imgs_hires[:, :, i],
                                                      dataformats="NCHW")
                    self.logger.experiment.add_images(key + "/lores",
                                                      imgs_lores[:, :, i],
                                                      dataformats="NCHW")
                    self.logger.experiment.add_images(key + "/mask",
                                                      masks[:, :, i],
                                                      dataformats="NCHW")


    def train_dataloader(self):
        """This method is called automatically by pytorch-lightning."""
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          sampler=self.train_sampler,
                          shuffle=self.train_sampler is None,
                          worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))

    def val_dataloader(self):
        """This method is called automatically by pytorch-lightning."""
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        """This method is called automatically by pytorch-lightning."""
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)

    def configure_optimizers(self):
        """This method is called automatically by pytorch-lightning."""
        params_dict = dict(self.named_parameters())
        weights = [
            v for k, v in params_dict.items()
            if "surv_head" not in k and "bias" not in k
        ]
        biases = [v for k, v in params_dict.items() if "bias" in k]
        mtlr_weights = self.surv_head.weight
        # Don't use weight decay on the biases and MTLR parameters, which have
        # their own separate L2 regularization
        optimizer = AdamW([
            {"params": weights},
            {"params": biases, "weight_decay": 0.},
            {"params": mtlr_weights, "weight_decay": self.hparams.mtlr_smooth_factor},
        ], lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = {
            "scheduler": MultiStepLR(optimizer, gamma=self.hparams.lr_decay_factor,
                                     milestones=self.hparams.lr_decay_milestones),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """Run a single training step on a batch of samples.

        This method is called automatically by pytorch-lightning.
        """
        output = self.forward((batch.image, batch.emr))
        surv_loss = self.surv_loss(output["surv"], batch.survival)
        total_loss = surv_loss
        log = {"training/surv/loss": surv_loss}
        out = {
            "pred_surv": output["surv"],
            "true_binary": batch.target_binary,
            "true_time": batch.survival_time,
            "true_event": batch.event,
        }
        log["training/total/loss"] = total_loss

        out["loss"] = total_loss
        out["log"] = log

        return out

    def training_epoch_end(self, outputs):
        """Compute performance metrics on the training dataset.

        This method is called automatically by pytorch-lightning.
        """
        pred_surv = torch.cat([x["pred_surv"] for x in outputs]).cpu()
        true_binary = torch.cat([x["true_binary"] for x in outputs]).cpu().numpy()
        true_time = torch.cat([x["true_time"] for x in outputs]).cpu().numpy()
        true_event = torch.cat([x["true_event"] for x in outputs]).cpu().numpy().astype(np.bool)

        two_year_bin = np.digitize(24, self.time_bins)
        survival_fn = mtlr_survival_at_times(pred_surv, np.pad(self.time_bins, (1, 0)), self.eval_times)
        pred_binary = 1 - mtlr_survival(pred_surv)[:, two_year_bin]
        roc_auc = roc_auc_score(true_binary, pred_binary)

        pred_risk = mtlr_risk(pred_surv).numpy()
        ci = concordance_index(true_time, -pred_risk, event_observed=true_event)

        log = {
            "training/surv/roc_auc_at_2yrs": roc_auc,
            "training/surv/ci": ci,
        }

        # log loss and metrics to Tensorboard
        loss_keys = [k for k in outputs[0]["log"].keys() if "/loss" in k]
        log.update({k: torch.stack([x["log"][k] for x in outputs]).mean() for k in loss_keys})
        return {
            "loss": log["training/total/loss"],
            "log": log
        }

    def validation_step(self, batch, batch_idx):
        """Run a single validation step on a batch of samples.

        This method is called automatically by pytorch-lightning.
        """
        output = self.forward((batch.image, batch.emr))
        surv_loss = self.surv_loss(output["surv"], batch.survival)
        total_loss = surv_loss
        out = {
            "pred_surv": output["surv"],
            "true_binary": batch.target_binary,
            "true_time": batch.survival_time,
            "true_event": batch.event,
            "validation/surv/loss": surv_loss
        }
        out["validation/total/loss"] = total_loss
        return out

    def validation_epoch_end(self, outputs):
        """Compute performance metrics on the validation dataset.

        This method is called automatically by pytorch-lightning.
        """
        pred_surv = torch.cat([x["pred_surv"] for x in outputs]).cpu()
        true_binary = torch.cat([x["true_binary"] for x in outputs]).cpu().numpy()
        true_time = torch.cat([x["true_time"] for x in outputs]).cpu().numpy()
        true_event = torch.cat([x["true_event"] for x in outputs]).cpu().numpy().astype(np.bool)

        log = {}
        two_year_bin = np.digitize(24, self.time_bins)
        survival_fn = mtlr_survival_at_times(pred_surv, np.pad(self.time_bins, (1, 0)), self.eval_times)
        pred_binary = 1 - mtlr_survival(pred_surv)[:, two_year_bin]
        roc_auc = roc_auc_score(true_binary, pred_binary)
        avg_prec = average_precision_score(true_binary, pred_binary)

        pred_risk = mtlr_risk(pred_surv).numpy()
        ci = concordance_index(true_time, -pred_risk, event_observed=true_event)
        ibs = integrated_brier_score(true_time, survival_fn, true_event, self.eval_times)

        if self.hparams.plot:
            pred_plot = plot_predictions(pred_surv[:16], true_time[:16],
                                        true_event[:16], np.pad(self.time_bins, (1, 0)))
            weights_plot = plot_weights(self.surv_head.weight.detach().cpu(), self.time_bins)

            self.logger.experiment.add_figure("predictions", pred_plot, global_step=self.global_step)
            self.logger.experiment.add_figure("weights", weights_plot, global_step=self.global_step)

        log = {
            "validation/surv/roc_auc_at_2yrs": roc_auc,
            "validation/surv/ap_at_2yrs": avg_prec,
            "validation/surv/ci": ci,
            "validation/surv/ibs": ibs
        }

        # log loss and metrics to Tensorboard
        loss_keys = [k for k in outputs[0].keys() if "loss" in k]
        log.update({k: torch.stack([x[k] for x in outputs]).mean() for k in loss_keys})
        self.logger.log_metrics(log, self.global_step)
        return {
            "val_loss": log["validation/total/loss"],
            "roc_auc": torch.tensor(log["validation/surv/roc_auc_at_2yrs"]),
            "log": log
        }

    def test_step(self, batch, batch_idx):
        """Run a single test step on a batch of samples.

        This method is called automatically by pytorch-lightning.
        """
        output = self.forward((batch.image, batch.emr))
        out = {
            "pred_surv": output["surv"],
        }
        return out

    def test_epoch_end(self, outputs):
        """Compute performance metrics on the test dataset.

        This method is called automatically by pytorch-lightning.
        """
        pred_surv = torch.cat([x["pred_surv"] for x in outputs]).detach().cpu()

        two_year_bin = np.digitize(24, self.time_bins)
        survival_fn = mtlr_survival_at_times(pred_surv, np.pad(self.time_bins, (1, 0)), self.eval_times)
        pred_binary = 1 - mtlr_survival(pred_surv)[:, two_year_bin]

        pred_risk = mtlr_risk(pred_surv).numpy()
        ids = self.test_dataset.emr_data.index
        res = pd.DataFrame({
            "Study ID": ids,
            "binary": pred_binary,
            "survival_event": pred_risk,
            **{f"survival_time_{t}": s for t, s in enumerate(survival_fn.T)}
        })
        res.to_csv(self.hparams.pred_save_path, index=False)
        return {}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        """Add model-specific hyperparameters to the parent parser."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size",
                            type=int,
                            default=16,
                            help="The batch size.")
        parser.add_argument("--lr",
                            type=float,
                            default=3e-4,
                            help="The initial learning rate.")
        parser.add_argument("--lr_decay_milestones",
                            type=int,
                            nargs="*",
                            default=[60, 160, 360],
                            help="Learning rate decay milestones (in epochs).")
        parser.add_argument("--lr_decay_factor",
                            type=float,
                            default=.1,
                            help="Learning rate decay factor (the LR will be multiplied by that number).")
        parser.add_argument("--weight_decay",
                            type=float,
                            default=1e-5,
                            help="The amount of weight decay to use.")
        parser.add_argument("--dropout",
                            type=float,
                            default=0.,
                            help="The dropout probability.")
        parser.add_argument("--patch_size",
                            type=int,
                            nargs=3,
                            default=(60, 60, 30),
                            help=("Size of the image patch extracted around "
                                  "each tumour."))
        parser.add_argument("--voxel_spacing",
                            type=float,
                            nargs=3,
                            default=(1., 1., 2.),
                            help=("Spacing of the image patch extracted around "
                                  "each tumour."))
        parser.add_argument("--lores_scale",
                            type=float,
                            default=.5)
        parser.add_argument("--balance_batches",
                            action="store_true",
                            help="Use balanced minibatch sampling.")
        parser.add_argument("--two_stream",
                            action="store_true",
                            help="Use two-stream network with downsampled context input.")
        parser.add_argument("--num_time_bins",
                            type=int,
                            default=40,
                            help="The number of 1 month-wide time bins to use.")
        parser.add_argument("--no_augmentation",
                            action="store_true",
                            help="Disable data augmentation.")
        parser.add_argument("--growth_rate",
                            type=int,
                            default=12,
                            help="DenseNet growth rate.")
        parser.add_argument("--use_emr_info",
                            action="store_true")
        parser.add_argument("--num_emr_features",
                            type=int,
                            default=20)
        parser.add_argument("--mtlr_smooth_factor",
                            type=float,
                            default=1e-5)
        parser.add_argument("--plot",
                            action="store_true")
        return parser
