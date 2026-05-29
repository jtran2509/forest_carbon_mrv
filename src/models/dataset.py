import os
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, 
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    ToTensord, 
    RandIntensityDistortiond
)

class ForestCarbonDataset(Dataset):
    """
    Custom Dataset for Loading Sentinel-2 chips and forest masks.
    Expect chpis in shape (4, 256, 256) -> B04, B03, B02, B08
    """
    def __init__(self, image_files, mask_files, transform=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image using Rasterio
        with rasterio.open(self.image_files[idx]) as src:
            image= src.read().astype(np.float32) # (4, 256, 256)

        # Load binary masks
        with rasterio.open(self.mask_files[idx]) as src:
            mask = src.read(1).astype(np.float32) # (256, 256)
            mask = np.expand_dims(mask, axis=0) # (1, 256, 256)

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
        RandIntensityDistortiond(keys=['image'], prob=0.2),

        # Convert to PyTorch Tensors
        ToTensord(keys =['image', 'label'])
    ])
