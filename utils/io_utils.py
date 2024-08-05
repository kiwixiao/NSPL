import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging

class MRIDataset(Dataset):
    def __init__(self, data_path, transform=None, img_size=(256, 256)):
        self.data_path = data_path
        self.transform = transform
        self.img_size = img_size
        self.images_dir = os.path.join(data_path, 'images')
        self.masks_dir = os.path.join(data_path, 'masks')
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])
        
        self.resize = transforms.Resize(img_size, Image.NEAREST)
        logging.info(f"Initialized MRIDataset with {len(self.image_files)} images, resize to {img_size}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image = self.resize(image)
        mask = self.resize(mask)

        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0

        if self.transform:
            image, mask = self.transform(image, mask)

        image = np.ascontiguousarray(image[np.newaxis, ...])
        mask = np.ascontiguousarray(mask[np.newaxis, ...])

        logging.debug(f"Loaded image {img_name}, shape: {image.shape}")
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float()

def load_dataset(data_path, img_size=(256, 256)):
    from preprocessing.data_preparation import augment_data
    return MRIDataset(data_path, transform=augment_data, img_size=img_size)