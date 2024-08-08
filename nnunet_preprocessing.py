import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import os
from PIL import Image

def resample_nifti(image, mask, target_spacing=(0.6, 0.6, 0.6)):
    # Load image and mask data
    image_data = image.get_fdata()
    mask_data = mask.get_fdata()
    original_affine = image.affine
    original_header = image.header

    # Get current spacing
    original_spacing = original_header.get_zooms()[:3]
    print(f"Original spacing: {original_spacing}")
    
    # Calculate the zoom factors
    zoom_factors = np.array(original_spacing) / np.array(target_spacing)
    print(f"Zoom factors: {zoom_factors}")

    # Resample the image and mask
    resampled_image = zoom(image_data, zoom_factors, order=3)
    resampled_mask = zoom(mask_data, zoom_factors, order=0)
    resampled_mask = (resampled_mask > 0.5).astype(np.uint8)
    
    # Update the affine matrix
    new_affine = original_affine.copy()
    for i in range(3):
        new_affine[i, i] = original_affine[i, i] * original_spacing[i] / target_spacing[i]
    
    return resampled_image, resampled_mask, new_affine, original_header

def pad_image(image, target_size=(256, 256)):
    width, height = image.size
    aspect_ratio = width / height
    target_aspect = target_size[0] / target_size[1]

    if aspect_ratio > target_aspect:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height), Image.BICUBIC if image.mode != 'L' else Image.NEAREST)

    padded_image = Image.new(image.mode, target_size, (0, 0, 0))
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    padded_image.paste(resized_image, (paste_x, paste_y))

    return padded_image

def preprocess_case(image_path, mask_path, target_spacing=(0.6, 0.6, 0.6), target_size=(128, 128), input_type='nifti'):
    if input_type == 'nifti':
        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)
        
        original_shape = image_nii.shape
        original_spacing = image_nii.header.get_zooms()[:3]
        
        resampled_image, resampled_mask, new_affine, original_header = resample_nifti(image_nii, mask_nii, target_spacing)
        
        print(f"Original image shape: {original_shape}, spacing: {original_spacing}")
        print(f"Resampled image shape: {resampled_image.shape}, new spacing: {target_spacing}")
        
        return resampled_image, resampled_mask, new_affine, original_header
    else:
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        
        original_shape = image.size
        
        padded_image = pad_image(image, target_size)
        padded_mask = pad_image(mask, target_size)
        
        print(f"Original image shape: {original_shape}")
        print(f"Padded image shape: {padded_image.size}")
        
        return padded_image, padded_mask, None, None

def save_nifti(data, affine, header, output_path):
    new_img = nib.Nifti1Image(data, affine)
    
    # Update header information
    new_img.header.set_data_shape(data.shape)
    new_img.header.set_zooms(np.sqrt(np.sum(affine[:3, :3]**2, axis=0)))
    
    # Copy over additional header information from the original header
    for key in header.keys():
        if key not in ['dim', 'pixdim']:
            new_img.header[key] = header[key]
    
    nib.save(new_img, output_path)
    print(f"Saved NIfTI file: {output_path}")
    print(f"Shape: {new_img.shape}, Spacing: {new_img.header.get_zooms()[:3]}")

def preprocess_dataset(input_dir, output_dir, target_spacing=(0.6, 0.6, 0.6), target_size=(128, 128), input_type='nifti'):
    os.makedirs(output_dir, exist_ok=True)
    
    if input_type == 'nifti':
        for file in os.listdir(input_dir):
            if file.startswith('OSAMRI') and file.endswith(('.nii', '.nii.gz')) and 'mask' not in file:
                image_path = os.path.join(input_dir, file)
                
                subject_id = file.split('_')[0]
                mask_files = [f for f in os.listdir(input_dir) if f.startswith(subject_id) and 'mask' in f]
                
                if not mask_files:
                    print(f"Mask not found for {file}")
                    continue
                
                mask_path = os.path.join(input_dir, mask_files[0])
                
                preprocessed_image, preprocessed_mask, new_affine, original_header = preprocess_case(image_path, mask_path, target_spacing=target_spacing, input_type=input_type)
                
                output_image_path = os.path.join(output_dir, f"preprocessed_{file}")
                output_mask_path = os.path.join(output_dir, f"preprocessed_{mask_files[0]}")
                
                save_nifti(preprocessed_image, new_affine, original_header, output_image_path)
                save_nifti(preprocessed_mask, new_affine, original_header, output_mask_path)
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
                
                padded_image, padded_mask, _, _ = preprocess_case(image_path, mask_path, target_size=target_size, input_type=input_type)
                
                output_image_path = os.path.join(output_dir, f"preprocessed_image_{file}")
                output_mask_path = os.path.join(output_dir, f"preprocessed_mask_{file}")
                
                padded_image.save(output_image_path)
                padded_mask.save(output_mask_path)
    
    print("Preprocessing completed.")

if __name__ == "__main__":
    input_dir = "path/to/input/directory"
    output_dir = "path/to/output/directory"
    target_spacing = (0.6, 0.6, 0.6)
    target_size = (128, 128)
    input_type = "nifti"  # or "png"
    preprocess_dataset(input_dir, output_dir, target_spacing=target_spacing, target_size=target_size, input_type=input_type)