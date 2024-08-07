import nibabel as nib
import numpy as np
from scipy import ndimage
import os
from PIL import Image

def resample_image(image, target_spacing, is_mask=False):
    if isinstance(image, nib.Nifti1Image):
        original_spacing = image.header.get_zooms()[:3]
        resize_factor = [o / t for o, t in zip(original_spacing, target_spacing)]
        new_shape = [int(s * f) for s, f in zip(image.shape, resize_factor)]
        
        if is_mask:
            return ndimage.zoom(image.get_fdata(), resize_factor, order=0)
        else:
            return ndimage.zoom(image.get_fdata(), resize_factor, order=3)
    else:
        # For PNG images, we don't have spacing information, so we just resize to target shape
        target_shape = (target_spacing[0], target_spacing[1])
        if is_mask:
            return np.array(image.resize(target_shape, Image.NEAREST))
        else:
            return np.array(image.resize(target_shape, Image.BICUBIC))

def normalize_intensity(image, mask=None):
    if mask is not None:
        mean = np.mean(image[mask > 0])
        std = np.std(image[mask > 0])
    else:
        mean = np.mean(image)
        std = np.std(image)
    
    return (image - mean) / std

def preprocess_case(image_path, mask_path, target_spacing=(1.5, 1.5, 2.0), input_type='nifti'):
    if input_type == 'nifti':
        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)
        
        resampled_image = resample_image(image_nii, target_spacing)
        resampled_mask = resample_image(mask_nii, target_spacing, is_mask=True)
        
        normalized_image = normalize_intensity(resampled_image, resampled_mask)
        
        return normalized_image, resampled_mask, image_nii.affine
    else:
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        resampled_image = resample_image(image, target_spacing[:2])
        resampled_mask = resample_image(mask, target_spacing[:2], is_mask=True)
        
        normalized_image = normalize_intensity(resampled_image, resampled_mask)
        
        return normalized_image, resampled_mask, None

def preprocess_dataset(input_dir, output_dir, input_type='nifti'):
    os.makedirs(output_dir, exist_ok=True)
    
    for image_file in os.listdir(input_dir):
        if (input_type == 'nifti' and (image_file.endswith('.nii') or image_file.endswith('.nii.gz'))) or \
           (input_type == 'png' and image_file.endswith('.png') and 'mask' not in image_file):
            image_path = os.path.join(input_dir, image_file)
            mask_file = image_file.replace('image', 'mask') if input_type == 'png' else image_file.replace('image', 'mask')
            mask_path = os.path.join(input_dir, mask_file)
            
            if not os.path.exists(mask_path):
                print(f"Mask not found for {image_file}")
                continue
            
            preprocessed_image, preprocessed_mask, affine = preprocess_case(image_path, mask_path, input_type=input_type)
            
            output_image_path = os.path.join(output_dir, f"preprocessed_{image_file}")
            output_mask_path = os.path.join(output_dir, f"preprocessed_{mask_file}")
            
            if input_type == 'nifti':
                nib.save(nib.Nifti1Image(preprocessed_image, affine), output_image_path)
                nib.save(nib.Nifti1Image(preprocessed_mask, affine), output_mask_path)
            else:
                Image.fromarray((preprocessed_image * 255).astype(np.uint8)).save(output_image_path)
                Image.fromarray(preprocessed_mask.astype(np.uint8)).save(output_mask_path)
    
    print("Preprocessing completed.")

if __name__ == "__main__":
    image_dir = "path/to/image/directory"
    output_dir = "path/to/output/directory"
    input_type = "nifti"  # or "png"
    preprocess_dataset(image_dir, output_dir, input_type)