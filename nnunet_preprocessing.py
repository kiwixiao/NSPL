import nibabel as nib
import numpy as np
from scipy import ndimage
import os

def resample_image(image, target_spacing, is_mask=False):
    original_spacing = image.header.get_zooms()[:3]
    resize_factor = [o / t for o, t in zip(original_spacing, target_spacing)]
    new_shape = [int(s * f) for s, f in zip(image.shape, resize_factor)]
    
    if is_mask:
        return ndimage.zoom(image.get_fdata(), resize_factor, order=0)
    else:
        return ndimage.zoom(image.get_fdata(), resize_factor, order=3)

def normalize_intensity(image, mask=None):
    if mask is not None:
        mean = np.mean(image[mask > 0])
        std = np.std(image[mask > 0])
    else:
        mean = np.mean(image)
        std = np.std(image)
    
    return (image - mean) / std

def preprocess_case(image_path, mask_path, target_spacing=(1.5, 1.5, 2.0)):
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    
    # Resample image and mask
    resampled_image = resample_image(image_nii, target_spacing)
    resampled_mask = resample_image(mask_nii, target_spacing, is_mask=True)
    
    # Normalize intensity
    normalized_image = normalize_intensity(resampled_image, resampled_mask)
    
    return normalized_image, resampled_mask, image_nii.affine

def preprocess_dataset(image_dir, mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.nii') or image_file.endswith('.nii.gz'):
            image_path = os.path.join(image_dir, image_file)
            mask_path = os.path.join(mask_dir, image_file.replace('image', 'mask'))
            
            if not os.path.exists(mask_path):
                print(f"Mask not found for {image_file}")
                continue
            
            preprocessed_image, preprocessed_mask, affine = preprocess_case(image_path, mask_path)
            
            output_image_path = os.path.join(output_dir, f"preprocessed_{image_file}")
            output_mask_path = os.path.join(output_dir, f"preprocessed_{image_file.replace('image', 'mask')}")
            
            nib.save(nib.Nifti1Image(preprocessed_image, affine), output_image_path)
            nib.save(nib.Nifti1Image(preprocessed_mask, affine), output_mask_path)
            
    print("Preprocessing completed.")

if __name__ == "__main__":
    image_dir = "path/to/image/directory"
    mask_dir = "path/to/mask/directory"
    output_dir = "path/to/output/directory"
    preprocess_dataset(image_dir, mask_dir, output_dir)