import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from PIL import Image

class NNUnetDataset(Dataset):
    def __init__(self, data_dir, target_size=(128, 128, 128), dimensions=3):
        self.data_dir = data_dir
        self.target_size = target_size
        self.dimensions = dimensions
        
        if dimensions == 3:
            self.image_files = [f for f in os.listdir(data_dir) if f.endswith('_mri.nii.gz') or f.endswith('_mri.nii')]
        else:
            self.image_dir = os.path.join(data_dir, 'images')
            self.mask_dir = os.path.join(data_dir, 'masks')
            self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)
    
    def resize_volume(self, image, is_mask=False):
        factors = [t / s for t, s in zip(self.target_size, image.shape)]
        order = 0 if is_mask else 3
        return zoom(image, factors, order=order)
    
    def pad_image(self, image):
        w, h = image.size
        target_w, target_h = self.target_size[:2]
        
        pad_w = max(target_w - w, 0)
        pad_h = max(target_h - h, 0)
        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
        
        padded_image = Image.new(image.mode, (max(w, target_w), max(h, target_h)), color=0)
        padded_image.paste(image, (padding[0], padding[1]))
        
        return padded_image
    
    def __getitem__(self, idx):
        if self.dimensions == 3:
            image_path = os.path.join(self.data_dir, self.image_files[idx])
            mask_path = image_path.replace('_mri.nii', '_mask.nii')
            
            image = nib.load(image_path).get_fdata()
            mask = nib.load(mask_path).get_fdata()
            
            image = self.resize_volume(image)
            mask = self.resize_volume(mask, is_mask=True)
            
            image = (image - image.mean()) / image.std()
            
            image = torch.from_numpy(image).float().unsqueeze(0)
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        else:
            image_path = os.path.join(self.image_dir, self.image_files[idx])
            mask_path = os.path.join(self.mask_dir, self.image_files[idx])
            
            image = Image.open(image_path).convert('L')
            mask = Image.open(mask_path).convert('L')
            
            image = self.pad_image(image)
            mask = self.pad_image(mask)
            
            image = np.array(image)
            mask = np.array(mask)
            
            image = (image - image.mean()) / (image.std() + 1e-8)
            
            image = torch.from_numpy(image).float().unsqueeze(0)
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        mask = (mask > 0).float()
        
        return {'image': image, 'mask': mask}