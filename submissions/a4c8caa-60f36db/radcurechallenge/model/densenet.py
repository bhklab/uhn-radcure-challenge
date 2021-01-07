"""3D DenseNet implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Iterable, Union


def bn_conv_relu(in_channels: int,
                 out_channels: int,
                 kernel_size: Union[Iterable[int], int],
                 stride: Union[Iterable[int], int],
                 padding: Union[Iterable[int], int]) -> nn.Module:
    """3D batch norm-conv-ReLU block."""
    return nn.Sequential(
        nn.BatchNorm3d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False))


def conv3x3x3(in_channels: int,
              out_channels: int,
              stride: Union[Iterable[int], int] = 1) -> nn.Module:
    """3x3x3 3D convolution block."""
    return bn_conv_relu(in_channels, out_channels,
                        kernel_size=3, stride=stride, padding=1)


def conv1x3x3(in_channels: int,
              out_channels: int,
              stride: Union[Iterable[int], int] = 1) -> nn.Module:
    """1x3x3 (in-plane) 3D convolution block."""
    return bn_conv_relu(in_channels, out_channels,
                        kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1))


def conv3x1x1(in_channels: int,
              out_channels: int,
              stride: Union[Iterable[int], int] = 1,
              z_pad: bool = True) -> nn.Module:
    """3x1x1 3D convolution block."""
    return bn_conv_relu(in_channels, out_channels,
                        kernel_size=(3, 1, 1), stride=stride, padding=(int(z_pad), 0, 0))


def conv1x1x1(in_channels: int,
              out_channels: int,
              stride: Union[Iterable[int], int] = 1) -> nn.Module:
    """1x1x1 3D convolution block."""
    return bn_conv_relu(in_channels, out_channels,
                        kernel_size=(1, 1, 1), stride=stride, padding=0)


class DenseBlock(nn.Module):
    """3D DenseNet building block."""
    def __init__(self,
                 in_channels: int,
                 growth_rate: int,
                 num_layers: int = 2,
                 memory_efficient: bool = True):
        """Initialize the module.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        growth_rate
            DenseNet growth rate.
        num_layers
            Number of convolutional layers within the block.
        memory_efficient
            Whether to use the memory-efficient implementation,
            trading off memory for compute time.
        """
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.memory_efficient = memory_efficient

        self.convs = nn.ModuleList()
        for i in range(num_layers * 3):
            if i % 3 == 2:
                conv = conv3x1x1
            else:
                conv = conv1x3x3
            self.convs.append(conv(self.in_channels + self.growth_rate * i,
                                   self.growth_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = [x]
        for conv in self.convs:
            if self.memory_efficient and any(x.requires_grad for x in features):
                out = checkpoint(conv, torch.cat(features, dim=1))
            else:
                out = conv(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """3D DenseNet transition layer."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout_p: float = 0.,
                 pool: bool = True):
        """Initialize the module.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        dropout_p
            Dropout probability.
        pool
            Whether to apply max-pooling after convolutions.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_p = dropout_p
        self.pool = pool

        self.conv = conv1x1x1(self.in_channels, self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.conv(x)
        if self.pool:
            out = F.max_pool3d(out, kernel_size=2, stride=2)
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        return out


class DenseNet3d(nn.Module):
    """3D DenseNet implementation.

    The block structure is adapted from previous work on retinal OCT images [1]_.
    The memory-efficient implementation is adapted from
    https://github.com/gpleiss/efficient_densenet_pytorch.

    References
    ----------
    ..[1] J. De Fauw et al., ‘Clinically applicable deep learning for diagnosis
    and referral in retinal disease’, Nat Med, vol. 24, no. 9, pp. 1342–1350,
    Sep. 2018.
    """
    def __init__(self,
                 growth_rate: int = 12,
                 dropout_p: float = 0.,
                 layer_config: Iterable[int] = (2, 2, 2, 3),
                 memory_efficient: bool = True):
        """Initialize the module.

        Parameters
        ----------
        dropout_p
            Dropout probability.
        growth_rate
            DenseNet growth rate.
        layer_config
            Number of convolutional layers within each block.
        memory_efficient
            Whether to use the memory-efficient implementation,
            trading off memory for compute time.
        """
        super().__init__()
        self.growth_rate = growth_rate
        self.dropout_p = dropout_p
        self.layer_config = layer_config
        self.memory_efficient = memory_efficient

        self.first_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 3, 3),
                      padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3),
                      padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.blocks = nn.Sequential()
        num_channels = 32
        for i, num_layers in enumerate(self.layer_config):
            self.blocks.add_module(f"block{i}",
                                   DenseBlock(num_channels,
                                              growth_rate=self.growth_rate,
                                              num_layers=num_layers,
                                              memory_efficient=self.memory_efficient))
            num_channels = num_channels + num_layers * 3 * self.growth_rate
            if i != len(self.layer_config) - 1:
                self.blocks.add_module(f"transition{i}",
                                       TransitionLayer(num_channels,
                                                       num_channels // 2,
                                                       dropout_p=self.dropout_p,
                                                       pool=True))
                num_channels //= 2

        self.out_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.first_conv(x)
        out = self.blocks(out)
        return out
