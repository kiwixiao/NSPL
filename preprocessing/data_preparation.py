import numpy as np
from scipy import ndimage
from sklearn.model_selection import KFold

def preprocess_mri(image):
    # the return is a numpy array
    
    # Normalize between 0 and 1
    image = (image - image.min()) / (image.max() - image.min())
    
    # Denoise (using Gaussian filter)
    image = ndimage.gaussian_filter(image, sigma=1)
    
    # Enhance contrast
    p2, p98 = np.percentile(image, (2, 98))
    image = np.clip(image, p2, p98)
    image = (image - p2) / (p98 - p2)
    
    return image

def augment_data(image, mask):
    # Random rotation
    angle = np.random.uniform(-20, 20)
    image = ndimage.rotate(image, angle, reshape=False, mode='nearest')
    mask = ndimage.rotate(mask, angle, reshape=False, mode='nearest')
    
    # Random flip
    if np.random.random() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    
    # Ensure arrays are contiguous
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)

def create_kfolds(dataset, n_splits=5, shuffle=True, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(kf.split(dataset))