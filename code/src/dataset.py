from torch.utils.data import Dataset
import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    ToTensord,
    RandAdjustContrastd
)

class ForestCarbonDataset(Dataset):
    """
    Custom Dataset for Loading Sentinel-2 chips and forest masks.
    Expect chpis in shape (2, 256, 256) -> Red, NIR
    """
    def __init__(self, image_files, mask_files, transform=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image using Rasterio (shape: 2, 256, 256) and mask (256, 256)
        image = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])

        # Add channel dimension to mask -> (1, 256, 256)
        mask = np.expand_dims(mask, axis=0)

        data_dict = {"image": image, "label": mask}

        # Apply MONAI transform
        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict

def get_train_transform():
    """
    Return a MONAI composition of augmentations to reach IoU > 0.85
    """
    return Compose([
        # Data augmentation
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=['image', 'label'], prob=0.75, max_k=3),

        # Intensity Augmentation (Handles atmospheric variance)
        RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.01),
        RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.3)),

        # Convert to PyTorch Tensors
        ToTensord(keys =['image', 'label'])
    ])
