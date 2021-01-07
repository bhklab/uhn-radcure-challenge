"""Augmentation transforms operating on SimpleITK images."""
from typing import Callable, List, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch


def find_centroid(mask: sitk.Image) -> np.ndarray:
    """Find the centroid of a binary image in image
    coordinates.

    Parameters
    ----------
    mask
        The bimary mask image.

    Returns
    -------
    np.ndarray
        The (x, y, z) coordinates of the centroid
        in image space.
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(mask, sitk.sitkUInt8))
    centroid_coords = stats.GetCentroid(1)
    return np.asarray(centroid_coords, dtype=np.float64)


def make_affine_transform(centre: List[float],
                          rotation: float = None,
                          flip: bool = False,
                          shear: float = None,
                          scaling: float = None,
                          translation: List[float] = None
                          ) -> sitk.Transform:
    """Builds an affine transform with the given parameters.

    Parameters
    ----------
    centre
        The transform centre in world coordinates.
    rotation
        The x-y rotation angle in radians.
    flip
        Whether to flip the image along the z-axis.
    shear
        The shear coefficient.
    scaling
        The scaling coefficient in the x-y plane.
    translation
        The translation vector.

    Returns
    -------
    sitk.Transform
        The constructed transform object.
    """
    transform = sitk.AffineTransform(3)
    transform.SetCenter(centre)
    if rotation:
        transform.Rotate(0, 1, rotation)
    if shear:
        transform.Shear(0, 1, shear)
    if scaling:
        transform.Scale((scaling, scaling, 1.))
    if translation:
        transform.SetTranslation(translation)
    if flip:
        matrix = list(transform.GetMatrix())
        matrix[0] *= -1
        transform.SetMatrix(matrix)
    return transform


def make_elastic_transform(ref_image: sitk.Image,
                           grid_size: List[float],
                           alpha: float,
                           spline_order: int = 3
                           ) -> sitk.Transform:
    """Builds a random elastic b-spline transform with the given parameters.

    Parameters
    ----------
    ref_image
        Reference image used to set up the grid geometry.
    grid_size
        The size of displacement grid.
    alpha
        The amount of random deformation.
    spline_order
        Order of the interpolating spline.

    Returns
    -------
    sitk.Transform
        The constructed transform object.
    """
    transform = sitk.BSplineTransformInitializer(ref_image, grid_size)
    params = np.asarray(transform.GetParameters())
    grid_size = np.asarray(grid_size)
    # the parameters are stored as a 1D array of length
    # ((grid_size_x + order) * (grid_size_y + order) * (grid_size_z + order) * n_dim
    # xy_size = ((grid_size[0] + spline_order) * (grid_size[1] + spline_order) + (grid_size[2] * spline_order)) * 2
    xy_size = np.prod(np.array(grid_size) + spline_order) * 2
    displacement = np.random.normal(0, alpha, xy_size)
    # we only want displacement in the (x,y) plane
    params[:xy_size] += displacement
    transform.SetParameters(params.tolist())
    return transform


def find_centre(image: sitk.Image) -> np.ndarray:
    """Find image centre in world coordinates.

    Parameters
    ----------
    image
        The input image.

    Returns
    -------
    np.ndarray
        The image centre in world coordinates.
    """
    centre_idx = np.array(image.GetSize()) / 2.
    centre_coords = image.TransformContinuousIndexToPhysicalPoint(centre_idx)
    return np.array(centre_coords)


class SpatialTransform:
    def __init__(self,
                 hires_size: List[float],
                 hires_spacing: List[float],
                 lores_size: List[float],
                 lores_spacing: List[float],
                 augment: bool = True,
                 rotation_range: List[float] = [-np.deg2rad(10), np.deg2rad(10)],
                 flip: bool = True,
                 shear_range: List[float] = [-.005, .005],
                 scaling_range: List[float] = [.8, 1.2],
                 translation_range: List[float] = [-10, 10],
                 elastic_grid_size: List[int] = [2, 2, 1],
                 elastic_alpha: float = 5):
        """Initialize the object.

        Most of the augmentation parameters can be safely left at the default
        values.

        Parameters
        ----------
        hires_size
            The size of the high-resolution image patch.
        hires_spacing
            The voxel spacing of the high-resolution image patch.
        lores_size
            The size of the low-resolution image patch.
        loes_spacing
            The voxel spacing of the low-resolution image patch.
        augment
            Whether to enable random data augmentation.
        rotation_range
            The minimum and maximum angle of rotation in radians.
        flip
            Whether to randomly flip the image along the z-axis.
        shear_range
            The minimum and maximum shear coefficients.
        scaling_range
            The minimum and maximum values of x-y scaling coefficient.
        translation_range
            The minimum and maximum translation values.
        elastic_grid_size
            The size of the grid used for random elastic deformation.
        elastic_alpha
            The magnitude of random elastic deformation.
        """
        self.lores_size = lores_size
        self.lores_spacing = lores_spacing
        self.hires_size = hires_size
        self.hires_spacing = hires_spacing
        self.augment = augment
        self.rotation_range = rotation_range
        self.flip = flip
        self.shear_range = shear_range
        self.scaling_range = scaling_range
        self.translation_range = translation_range
        self.elastic_grid_size = elastic_grid_size
        self.elastic_alpha = elastic_alpha

    def __call__(self,
                 image: sitk.Image,
                 mask: sitk.Image) -> Tuple[sitk.Image, sitk.Image, sitk.Image]:
        """Transform the image and mask with optional random augmentation.

        Only the high-resolution mask is generated.

        Parameters
        ----------
        image
            The image to transform.
        mask
            Binary mask defining the region of interest.

        Returns
        -------
        tuple of sitk.Image
            The transformed high and low-resolution images and
            the high-resolution mask.
        """
        hires_ref = sitk.Image(*self.hires_size, sitk.sitkFloat32)
        lores_ref = sitk.Image(*self.lores_size, sitk.sitkFloat32)
        hires_ref.SetSpacing(self.hires_spacing)
        lores_ref.SetSpacing(self.lores_spacing)

        image_centre = find_centre(image)
        hires_centre = find_centre(hires_ref)
        lores_centre = find_centre(lores_ref)

        hires_ref.SetOrigin(image_centre - hires_centre)
        lores_ref.SetOrigin(image_centre - lores_centre)

        transform = sitk.Transform(3, sitk.sitkComposite)
        mask_centroid = find_centroid(mask)
        centering = sitk.TranslationTransform(3, (mask_centroid - image_centre).tolist())
        transform.AddTransform(centering)
        if self.augment:
            rotation = np.random.uniform(*self.rotation_range) if self.rotation_range is not None else None
            flip = np.random.binomial(1, .5) if self.flip else None
            shear = np.random.uniform(*self.shear_range) if self.shear_range is not None else None
            scaling = np.random.uniform(*self.scaling_range) if self.scaling_range is not None else None
            translation = [
                np.random.uniform(*self.translation_range),
                np.random.uniform(*self.translation_range),
                0.
            ] if self.translation_range is not None else None
            transform.AddTransform(make_affine_transform(image_centre, rotation, flip, shear, scaling, translation))
            if self.elastic_alpha is not None and self.elastic_grid_size is not None:
                transform.AddTransform(make_elastic_transform(hires_ref, self.elastic_grid_size, self.elastic_alpha))

        fill_val = float(sitk.GetArrayViewFromImage(image).min())

        hires_image = sitk.Resample(image, hires_ref, transform, sitk.sitkLinear, fill_val)
        lores_image = sitk.Resample(image, lores_ref, transform, sitk.sitkLinear, fill_val)
        hires_mask = sitk.Resample(mask, hires_ref, transform, sitk.sitkNearestNeighbor, 0)

        return hires_image, lores_image, hires_mask


class ToTensor:
    """Convert one or multiple SimpleITK images to torch.Tensor."""
    def _image_to_tensor(self, image: sitk.Image) -> torch.Tensor:
        array = sitk.GetArrayFromImage(image)
        tensor = torch.from_numpy(array).unsqueeze(0).float()
        return tensor

    def __call__(self, *args: sitk.Image) -> List[torch.Tensor]:
        """Apply the transform.

        Parameters
        ----------
        image
            Image to convert to tensor.

        Returns
        -------
        torch.Tensor or list of torch.Tensor
            The converted tensor(s).
        """
        return [self._image_to_tensor(x) for x in args]

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class RandomNoise:
    """Add zero-mean Gaussian noise to an image."""
    def __init__(self, stds: float):
        """Initialize the transform.

        Parameters
        ----------
        std
            The standard deviation of noise.
        """
        self.stds = stds

    def __call__(self, *args: sitk.Image) -> List[sitk.Image]:
        """Apply the transform.

        Parameters
        ----------
        image
            Image to transform.

        Returns
        -------
        sitk.Image
            The transformed image.
        """
        out = []
        for x, std in zip(args, self.stds):
            if std > 0:
                noise = np.random.normal(0, std, x.GetSize()[::-1]).astype(np.float32)
                noise = sitk.GetImageFromArray(noise)
                noise.CopyInformation(x)
                out.append(x + noise)
            else:
                out.append(x)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(stds={self.stds})"


class Normalize:
    """Normalize an image by subtracting the dataset mean and dividing by the
    dataset standard deviation.
    """
    def __init__(self, means: float, stds: float):
        """Initialize the transform.

        Parameters
        ----------
        mean
            The dataset mean.
        std
            The dataset standard deviation.
        """
        self.means = means
        self.stds = stds

    def __call__(self, *args: sitk.Image) -> List[sitk.Image]:
        """Apply the transform.

        Parameters
        ----------
        image
            Image to transform.

        Returns
        -------
        sitk.Image
            The transformed image.
        """
        return [sitk.Cast((x - mean) / std, sitk.sitkFloat32) for x, mean, std in zip(args, self.means, self.stds)]

    def __repr__(self):
        return f"{self.__class__.__name__}(means={self.means}, stds={self.stds})"


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            if isinstance(x, (tuple, list)):
                x = t(*x)
            else:
                x = t(x)
        return x

    def __repr__(self):
        transform_reprs = '\n'.join(["    " + t.__repr__() + "," for t in self.transforms])
        return f"{self.__class__.__name__}([\n{transform_reprs}\n])"


class ColumnNormalize:
    """Normalize columns of a tensor to zero mean and unit variance."""
    def __init__(self,
                 column_means: Union[List[float], np.ndarray],
                 column_stds: Union[List[float], np.ndarray]):
        self.column_means = torch.tensor(column_means, dtype=torch.float)
        self.column_stds = torch.tensor(column_stds, dtype=torch.float)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.column_means) / self.column_stds
