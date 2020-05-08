"""Augmentation transforms operating on 3D tensors."""

import numpy as np
import SimpleITK as sitk
import torch


class ToTensor:
    def __call__(self, image):
        array = sitk.GetArrayFromImage(image)
        tensor = torch.from_numpy(array).unsqueeze(0).float()
        return tensor


class RandomInPlaneRotation:
    def __init__(self, max_angle, fill_value=-1024.):
        self.max_angle = max_angle
        self.fill_value = fill_value

    def __call__(self, x):
        angle = -self.max_angle + 2 * self.max_angle * torch.rand(1).item()
        rotation_centre = np.array(x.GetSize()) / 2
        rotation_centre = x.TransformContinuousIndexToPhysicalPoint(rotation_centre)

        rotation = sitk.Euler3DTransform(
            rotation_centre,
            0,      # the angle of rotation around the x-axis, in radians -> coronal rotation
            0,      # the angle of rotation around the y-axis, in radians -> saggittal rotation
            angle,  # the angle of rotation around the z-axis, in radians -> axial rotation
            (0., 0., 0.)  # no translation
        )
        return sitk.Resample(x, x, rotation, sitk.sitkLinear, self.fill_value)


class RandomFlip:
    def __init__(self, dim):
        self.dim = dim
        self.flip_mask = [i == self.dim for i in range(3)]

    def __call__(self, x):
        if np.random.random() > .5:
            x = sitk.Flip(x, self.flip_mask)
        return x


class RandomNoise:
    def __init__(self, std=1.):
        self.std = std

    def __call__(self, x):
        # use Pytorch random generator for consistent use of seeds
        noise = (torch.randn(x.GetSize()[::-1]) * self.std).numpy()
        noise = sitk.GetImageFromArray(noise)
        noise.CopyInformation(x)
        return x + noise


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x = (x - self.mean) / self.std
        return sitk.Cast(x, sitk.sitkFloat32)
