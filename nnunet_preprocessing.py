import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import os
from PIL import Image

def resample_nifti_isotropic(volume, is_mask=False):
    # Get current spacing and shape
    current_spacing = np.array(volume.header.get_zooms()[:3])
    current_shape = np.array(volume.shape[:3])
    
    # Determine the new isotropic spacing (smallest current spacing)
    new_spacing = np.min(current_spacing)
    
    # Calculate scale factors
    scale_factors = current_spacing / new_spacing
    
    # Calculate new shape
    new_shape = np.round(current_shape * scale_factors).astype(int)
    
    # Resample
    if is_mask:
        resampled = zoom(volume.get_fdata(), scale_factors, order=0, mode='nearest')
        resampled = (resampled > 0.5).astype(np.uint8)
    else:
        resampled = zoom(volume.get_fdata(), scale_factors, order=3, mode='constant')
    
    print(f"Original shape: {current_shape}, spacing: {current_spacing}")
    print(f"New shape: {new_shape}, spacing: {new_spacing}")
    
    return resampled, np.array([new_spacing, new_spacing, new_spacing])


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

def preprocess_case(image_path, mask_path, target_size=(128, 128), input_type='nifti'):
    if input_type == 'nifti':
        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)
        
        resampled_image, achieved_spacing = resample_nifti_isotropic(image_nii)
        resampled_mask, _ = resample_nifti_isotropic(mask_nii, is_mask=True)
        
        return resampled_image, resampled_mask, image_nii, mask_nii, achieved_spacing
    else:
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        
        original_shape = image.size
        
        padded_image = pad_image(image, target_size)
        padded_mask = pad_image(mask, target_size)
        
        print(f"Original image shape: {original_shape}")
        print(f"Padded image shape: {padded_image.size}")
        
        return padded_image, padded_mask, None, None

def save_nifti(data, original_img, new_spacing, filename):
    # Create a new affine matrix with the new spacing
    new_affine = original_img.affine.copy()
    for i in range(3):
        new_affine[i, i] = new_spacing[i] * np.sign(new_affine[i, i])

    # Create a new NIfTI image
    new_img = nib.Nifti1Image(data, new_affine)
    
    # Update header information
    new_header = new_img.header
    new_header.set_zooms(new_spacing)
    new_header.set_data_dtype(data.dtype)
    
    # Copy relevant header fields from the original image
    for field in ['descrip', 'intent_name', 'qform_code', 'sform_code']:
        if field in original_img.header:
            setattr(new_header, field, original_img.header[field])
    
    # Update qform and sform
    qform_code = int(original_img.header.get('qform_code'))
    sform_code = int(original_img.header.get('sform_code'))
    new_img.set_qform(new_affine, code=qform_code)
    new_img.set_sform(new_affine, code=sform_code)
    
    # Ensure pixdim is set correctly
    new_header['pixdim'][1:4] = new_spacing
    
    # Save the image
    nib.save(new_img, filename)
    print(f"Saved NIfTI file: {filename}")
    print(f"New shape: {data.shape}, New spacing: {new_spacing}")

    # Verify the saved file
    loaded_img = nib.load(filename)
    print(f"Loaded shape: {loaded_img.shape}, Loaded spacing: {loaded_img.header.get_zooms()[:3]}")

def preprocess_dataset(input_dir, output_dir, target_size=(128, 128), input_type='nifti'):
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
                
                resampled_image, resampled_mask, image_nii, mask_nii, achieved_spacing = preprocess_case(
                    image_path, mask_path, target_size=target_size, input_type=input_type
                )
                
                output_image_path = os.path.join(output_dir, f"preprocessed_{file}")
                output_mask_path = os.path.join(output_dir, f"preprocessed_{mask_files[0]}")
                
                save_nifti(resampled_image, image_nii, achieved_spacing, output_image_path)
                save_nifti(resampled_mask, mask_nii, achieved_spacing, output_mask_path)
                # Verify saved files
                verify_img = nib.load(output_image_path)
                verify_mask = nib.load(output_mask_path)
                print(f"Verified image spacing: {verify_img.header.get_zooms()[:3]}")
                print(f"Verified mask spacing: {verify_mask.header.get_zooms()[:3]}")
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
                
                padded_image, padded_mask, _, _ = preprocess_case(
                    image_path, mask_path, target_size=target_size, input_type=input_type
                )
                
                output_image_path = os.path.join(output_dir, f"preprocessed_image_{file}")
                output_mask_path = os.path.join(output_dir, f"preprocessed_mask_{file}")
                
                padded_image.save(output_image_path)
                padded_mask.save(output_mask_path)
    
    print("Preprocessing completed.")

if __name__ == "__main__":
    input_dir = "path/to/input/directory"
    output_dir = "path/to/output/directory"
    target_size = (128, 128)
    input_type = "nifti"  # or "png"
    preprocess_dataset(input_dir, output_dir, target_size=target_size, input_type=input_type)