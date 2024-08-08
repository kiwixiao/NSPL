import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import os
from PIL import Image

def resample_nifti_isotropic(volume, is_mask=False):
    current_spacing = np.array(volume.header.get_zooms()[:3])
    current_shape = np.array(volume.shape[:3])
    new_spacing = np.min(current_spacing)
    scale_factors = current_spacing / new_spacing
    new_shape = np.round(current_shape * scale_factors).astype(int)
    if is_mask:
        resampled = zoom(volume.get_fdata(), scale_factors, order=0, mode='nearest')
        resampled = (resampled > 0.5).astype(np.uint8)
    else:
        resampled = zoom(volume.get_fdata(), scale_factors, order=3, mode='constant')
    return resampled, np.array([new_spacing, new_spacing, new_spacing])

def pad_image(image, target_size=(64, 64)):
    w, h = image.size
    target_w, target_h = target_size
    pad_w = max(target_w - w, 0)
    pad_h = max(target_h - h, 0)
    padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
    padded_image = Image.new(image.mode, (max(w, target_w), max(h, target_h)), color=0)
    padded_image.paste(image, (padding[0], padding[1]))
    return padded_image

def preprocess_case(image_path, mask_path, target_size=(128, 128, 128), input_type='nifti'):
    if input_type == 'nifti':
        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)
        resampled_image, achieved_spacing = resample_nifti_isotropic(image_nii)
        resampled_mask, _ = resample_nifti_isotropic(mask_nii, is_mask=True)
        return resampled_image, resampled_mask, image_nii.affine, achieved_spacing
    else:
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        original_shape = image.size
        padded_image = pad_image(image, target_size[:2])
        padded_mask = pad_image(mask, target_size[:2])
        return padded_image, padded_mask, None, None

def save_nifti(data, original_img, new_spacing, filename):
    new_affine = original_img.affine.copy()
    for i in range(3):
        new_affine[i, i] = new_spacing[i] * np.sign(new_affine[i, i])
    new_img = nib.Nifti1Image(data, new_affine)
    new_header = new_img.header
    new_header.set_zooms(new_spacing)
    new_header.set_data_dtype(data.dtype)
    for field in ['descrip', 'intent_name', 'qform_code', 'sform_code']:
        if field in original_img.header:
            setattr(new_header, field, original_img.header[field])
    qform_code = int(original_img.header.get('qform_code'))
    sform_code = int(original_img.header.get('sform_code'))
    new_img.set_qform(new_affine, code=qform_code)
    new_img.set_sform(new_affine, code=sform_code)
    new_header['pixdim'][1:4] = new_spacing
    nib.save(new_img, filename)

def preprocess_dataset(input_dir, output_dir, target_size=(128, 128, 128), input_type='nifti'):
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
                resampled_image, resampled_mask, affine, achieved_spacing = preprocess_case(
                    image_path, mask_path, target_size=target_size, input_type=input_type
                )
                output_image_path = os.path.join(output_dir, f"preprocessed_{file}")
                output_mask_path = os.path.join(output_dir, f"preprocessed_{mask_files[0]}")
                save_nifti(resampled_image, nib.load(image_path), achieved_spacing, output_image_path)
                save_nifti(resampled_mask, nib.load(mask_path), achieved_spacing, output_mask_path)
    else:
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
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
                    image_path, mask_path, target_size=target_size[:2], input_type=input_type
                )
                output_image_path = os.path.join(output_dir, 'images', f"preprocessed_{file}")
                output_mask_path = os.path.join(output_dir, 'masks', f"preprocessed_{file}")
                padded_image.save(output_image_path)
                padded_mask.save(output_mask_path)
    
    print("Preprocessing completed.")

if __name__ == "__main__":
    input_dir = "path/to/input/directory"
    output_dir = "path/to/output/directory"
    target_size = (128, 128, 128)
    input_type = "nifti"  # or "png"
    preprocess_dataset(input_dir, output_dir, target_size=target_size, input_type=input_type)