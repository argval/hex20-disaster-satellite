import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class BurnAreaDataset(Dataset):
    """Custom PyTorch Dataset for loading Sentinel-2 patches and burn masks."""

    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with all the input image patches.
            mask_dir (str): Directory with all the corresponding mask patches.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get a sorted list of image filenames to ensure alignment
        self.ids = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = self.ids[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        try:
            with rasterio.open(img_path) as src:
                # Sentinel-2 data is often uint16. Read and convert to float32.
                image = src.read().astype(np.float32)

            with rasterio.open(mask_path) as src:
                # Mask is single-band. Read and ensure it's float32 for loss calculation.
                mask = src.read(1).astype(np.float32)
                # Add a channel dimension to make it (1, H, W) for consistency
                mask = np.expand_dims(mask, axis=0)

        except rasterio.errors.RasterioIOError as e:
            print(f"Error opening file: {e}")
            # Return a dummy sample or handle error as appropriate
            return None, None

        # Basic normalization for Sentinel-2 bands. 
        # A more robust method would involve calculating mean/std from the dataset,
        # but scaling to [0, 1] is a good starting point for an MVP.
        # We clip to avoid extreme values from sensor artifacts.
        image = np.clip(image, 0, 10000) / 10000.0

        # Convert numpy arrays to PyTorch tensors
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        if self.transform:
            # Here you could apply data augmentation
            # For now, we pass
            pass

        return image_tensor, mask_tensor
