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
        target_shape = (int(target_spacing[0]), int(target_spacing[1]))
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

def preprocess_case(image_path, mask_path, target_spacing=(128, 128, 128), input_type='nifti'):
    if input_type == 'nifti':
        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)
        
        original_shape = image_nii.shape
        original_mask = mask_nii.get_fdata()
        original_positive_pixels = np.sum(original_mask > 0)
        
        resampled_image = resample_image(image_nii, target_spacing)
        resampled_mask = resample_image(mask_nii, target_spacing, is_mask=True)
        
        affine = image_nii.affine
    else:
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        original_shape = np.array(image).shape
        original_mask = np.array(mask)
        original_positive_pixels = np.sum(original_mask > 0)
        
        resampled_image = resample_image(image, target_spacing[:2])
        resampled_mask = resample_image(mask, target_spacing[:2], is_mask=True)
        
        affine = None
    
    # Ensure both resampled image and mask have the same shape
    target_shape = resampled_image.shape
    if resampled_mask.shape != target_shape:
        resampled_mask = resize_to_match(resampled_mask, target_shape)
    
    resampled_positive_pixels = np.sum(resampled_mask > 0)
    pixel_ratio = resampled_positive_pixels / original_positive_pixels
    print(f"Original positive pixels: {original_positive_pixels}")
    print(f"Resampled positive pixels: {resampled_positive_pixels}")
    print(f"Ratio of positive pixels (resampled / original): {pixel_ratio:.4f}")

    if abs(pixel_ratio - 1) > 0.1:
        print("WARNING: Significant change in the number of positive pixels after resampling.")
        print("Consider adjusting the target spacing or checking the preprocessing steps.")
    
    normalized_image = normalize_intensity(resampled_image, resampled_mask)
    
    print(f"Original image shape: {original_shape}")
    print(f"Resampled image shape: {resampled_image.shape}")
    print(f"Resampled mask shape: {resampled_mask.shape}")
    
    return normalized_image, resampled_mask, affine

def resize_to_match(array, target_shape):
    return ndimage.zoom(array, np.array(target_shape) / np.array(array.shape), order=0)

def preprocess_dataset(input_dir, output_dir, input_type='nifti'):
    os.makedirs(output_dir, exist_ok=True)
    
    if input_type == 'nifti':
        for file in os.listdir(input_dir):
            if file.startswith('OSAMRI') and 'mri' in file and file.endswith(('.nii', '.nii.gz')):
                image_path = os.path.join(input_dir, file)
                subject_id = file.split('_')[0]
                
                mask_file = next((f for f in os.listdir(input_dir) if f.startswith(subject_id) and 'mask' in f), None)
                
                if mask_file is None:
                    print(f"Mask not found for {file}")
                    continue
                
                mask_path = os.path.join(input_dir, mask_file)
                
                preprocessed_image, preprocessed_mask, affine = preprocess_case(image_path, mask_path, input_type=input_type)
                
                output_image_path = os.path.join(output_dir, f"preprocessed_{file}")
                output_mask_path = os.path.join(output_dir, f"preprocessed_{mask_file}")
                
                nib.save(nib.Nifti1Image(preprocessed_image, affine), output_image_path)
                nib.save(nib.Nifti1Image(preprocessed_mask, affine), output_mask_path)
    else:
        image_dir = os.path.join(input_dir, 'images')
        mask_dir = os.path.join(input_dir, 'masks')
        
        for file in os.listdir(image_dir):
            if file.endswith('.png'):
                image_path = os.path.join(image_dir, file)
                mask_path = os.path.join(mask_dir, file)
                
                if not os.path.exists(mask_path):
                    print(f"Mask not found for {file}")
                    continue
                
                preprocessed_image, preprocessed_mask, _ = preprocess_case(image_path, mask_path, input_type=input_type)
                
                output_image_path = os.path.join(output_dir, f"preprocessed_image_{file}")
                output_mask_path = os.path.join(output_dir, f"preprocessed_mask_{file}")
                
                Image.fromarray((preprocessed_image * 255).astype(np.uint8)).save(output_image_path)
                Image.fromarray(preprocessed_mask.astype(np.uint8)).save(output_mask_path)
    
    print("Preprocessing completed.")

if __name__ == "__main__":
    input_dir = "path/to/input/directory"
    output_dir = "path/to/output/directory"
    input_type = "nifti"  # or "png"
    preprocess_dataset(input_dir, output_dir, input_type)