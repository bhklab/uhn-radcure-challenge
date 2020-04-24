"""Augmentation transforms operating on 3D tensors."""

import numpy as np
import SimpleITK as sitk
import torch
from torchvision.transforms import *


class ToTensor:
    def __call__(self, image):
        array = sitk.GetArrayFromImage(image)
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor


class Random90DegRotation:
    def __call__(self, tensor):
        k = np.random.randint(0, 4, 1)
        return torch.rot90(tensor, k, (1, 2))


class RandomFlip:
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, tensor):
        if np.random.random() > .5:
            tensor = torch.flip(tensor, self.dims)
        return tensor


class RandomNoise:
    def __init__(self, std=1.):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std
        return tensor + noise


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return (tensor - mean) / std
