import numpy as np 
from patchify import patchify 

class SatelliteChipper:
    def __init__(self, patch_size=256, stride=256):
        self.patch_size = patch_size
        self.stride = stride

    def create_chips(self, image_array):
        """
        Slices a large multi-spectral image into smaller patches.

        Args:
            image_array: Shape (Channels, Height, Width)

        Returns:
            np.ndarray: Array of patches with shape (N, Channels, 256, 256)
        """
        channels, h, w = image_array.shape

        # Ensure the image dimension are divisible by patch_size
        # Padding might be required if not divisible
        new_h = (h // self.patch_size) * self.patch_size
        new_w = (w // self.patch_size) * self.patch_size
        image_trimmed = image_array[:, :new_h, :new_w]

        patches = []

        # Iterate through channels to use patchify
        for c in range(channels):
            #Patchify expects (H, W) so we pass one channel at a time
            channel_patches = patchify(image_trimmed[c], (self.patch_size, self.patch_size), step=self.stride)
            patches.append(channel_patches)

        # Reshape (Channels, n_h, n_w, 256, 256) -> (N, channels, 256, 256)
        patches = np.array(patches)
        n_h, n_w = patches.shape[1], patches.shape[2]
        patches = patches.transpose(1, 2, 0, 3, 4).reshape(-1, channels, self.patch_size, self.patch_size)

        return patches
    def filter_empty_patches(self, patches, threshold=0.1):
        """
        Removes patches that are mostly "no data" or black pixels
        """
        valid_patches = []
        for p in patches:
            # Check if the percentage of zero-pixels is below threshold
            if np.mean(p==0) < threshold:
                valid_patches.append(p)

        return np.array(valid_patches)
    
    