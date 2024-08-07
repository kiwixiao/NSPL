# nnunet_dataset.py

import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

class NNUnetDataset(Dataset):
    def __init__(self, data_dir, target_size=(128, 128, 128)):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('_mri.nii.gz') or f.endswith('_mri.nii')]
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_files)
    
    def resize_volume(self, image, is_mask=False):
        # Calculate resize factors
        factors = [t / s for t, s in zip(self.target_size, image.shape)]
        
        # Use nearest neighbor interpolation for masks, and trilinear for images
        order = 0 if is_mask else 3
        
        return zoom(image, factors, order=order)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        mask_path = image_path.replace('_mri.nii*', '_mask.nii*')
        
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        
        # Resize image and mask
        image = self.resize_volume(image)
        mask = self.resize_volume(mask, is_mask=True)
        
        # Normalize image
        image = (image - image.mean()) / image.std()
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        #print(f'Shape of image load from numpy {np.shape(image)}')
        
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        #print(f"Shape of mask after load using torch.from_numpy method {np.shape(mask)}")
        
        # Ensure mask is binary (0 or 1)
        mask = (mask > 0).float()
        
        return {'image': image, 'mask': mask}