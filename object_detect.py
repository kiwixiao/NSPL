import os
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import glob
import torch
from scipy.ndimage import zoom

def find_objects(mask, margin=1.2):
    _, binary_mask = cv2.threshold(mask.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate center
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Apply margin
        w_with_margin = w * margin
        h_with_margin = h * margin
        
        # Ensure the box doesn't exceed image boundaries
        x_min = max(0, center_x - w_with_margin / 2)
        y_min = max(0, center_y - h_with_margin / 2)
        x_max = min(mask.shape[1], center_x + w_with_margin / 2)
        y_max = min(mask.shape[0], center_y + h_with_margin / 2)
        
        # Recalculate width and height
        w_final = x_max - x_min
        h_final = y_max - y_min
        
        # Convert to YOLO format
        center_x = (x_min + w_final / 2) / mask.shape[1]
        center_y = (y_min + h_final / 2) / mask.shape[0]
        width = w_final / mask.shape[1]
        height = h_final / mask.shape[0]
        
        objects.append([0, center_x, center_y, width, height])  # 0 is the class index for airway
    
    return objects

def save_image(img, filename):
    # Normalize to [0, 1]
    img_normalized = (img - img.min()) / (img.max() - img.min())
    
    # Standardize to zero mean and unit variance
    img_standardized = (img_normalized - np.mean(img_normalized)) / np.std(img_normalized)
    
    # Rescale to [0, 255] for saving as PNG
    img_rescaled = ((img_standardized - img_standardized.min()) / (img_standardized.max() - img_standardized.min()) * 255).astype(np.uint8)
    
    Image.fromarray(img_rescaled).save(filename)
    print(f"Saved normalized and standardized image: {filename}")

def save_yolo_annotation(objects, filename):
    with open(filename, 'w') as f:
        for obj in objects:
            f.write(' '.join(map(str, obj)) + '\n')
    print(f"Saved annotation: {filename}")

def resample_volume(volume, target_shape=(256, 256, 256), target_spacing=(0.6, 0.6, 0.6)):
    current_spacing = volume.header.get_zooms()
    scale_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
    resampled = zoom(volume.get_fdata(), scale_factors, order=3)  # order=3 for linear interpolation
    
    # Pad or crop to exactly 256x256x256
    pad_width = [(max(t - s, 0) // 2, max(t - s, 0) - max(t - s, 0) // 2) for t, s in zip(target_shape, resampled.shape)]
    cropped = resampled[
        max(0, (resampled.shape[0] - target_shape[0]) // 2):min(resampled.shape[0], (resampled.shape[0] + target_shape[0]) // 2),
        max(0, (resampled.shape[1] - target_shape[1]) // 2):min(resampled.shape[1], (resampled.shape[1] + target_shape[1]) // 2),
        max(0, (resampled.shape[2] - target_shape[2]) // 2):min(resampled.shape[2], (resampled.shape[2] + target_shape[2]) // 2)
    ]
    padded = np.pad(cropped, pad_width, mode='constant')
    
    return padded

def process_mri_to_yolo(mri_path, mask_path, output_dir, subject_id):
    print(f"Processing MRI: {mri_path}")
    print(f"Processing Mask: {mask_path}")
    
    # Load and reorient MRI and mask
    mri_img = nib.load(mri_path)
    mask_img = nib.load(mask_path)
    
    mri_img = nib.as_closest_canonical(mri_img)
    mask_img = nib.as_closest_canonical(mask_img)
    
    # Resample to 256x256x256 with 0.6mm spacing
    mri_resampled = resample_volume(mri_img)
    mask_resampled = resample_volume(mask_img)
    
    # Binarize the mask
    mask_resampled = (mask_resampled > 0).astype(np.float32)
    
    print(f"Resampled MRI shape: {mri_resampled.shape}")
    print(f"Resampled Mask shape: {mask_resampled.shape}")
    
    # Process coronal slices
    for i in range(mri_resampled.shape[1]):  # Coronal slices
        slice_img = mri_resampled[:,i,:]
        slice_mask = mask_resampled[:,i,:]
        
        objects = find_objects(slice_mask)
        
        if objects:
            img_filename = f"{output_dir}/images/{subject_id}_{i:03d}.png"
            save_image(slice_img, img_filename)
            
            txt_filename = f"{output_dir}/labels/{subject_id}_{i:03d}.txt"
            save_yolo_annotation(objects, txt_filename)
        else:
            print(f"No objects found in coronal slice {i} of {subject_id}")
    
    print(f"Processed all coronal slices for {subject_id}")

def process_all_mri_data(image_dir, mask_dir, output_dir):
    print(f"Processing all MRI data:")
    print(f"Image directory: {image_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Output directory: {output_dir}")

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    print(f"Contents of image directory:")
    print(os.listdir(image_dir))
    print(f"Contents of mask directory:")
    print(os.listdir(mask_dir))

    # Find all MRI files
    mri_files = glob.glob(os.path.join(image_dir, "OSAMRI*_mri.nii")) + glob.glob(os.path.join(image_dir, "OSAMRI*_mri.nii.gz"))

    for mri_file in mri_files:
        print(f"Processing MRI file: {mri_file}")
        
        # Extract subject ID
        subject_id = os.path.basename(mri_file).split('_')[0]
        
        # Find corresponding mask file
        mask_pattern = os.path.join(mask_dir, f"{subject_id}*_mask.nii*")
        mask_files = glob.glob(mask_pattern)
        
        if mask_files:
            mask_file = mask_files[0]  # Take the first matching mask file
            print(f"Found corresponding mask: {mask_file}")
            process_mri_to_yolo(mri_file, mask_file, output_dir, subject_id)
        else:
            print(f"Warning: No matching mask found for {mri_file}")
            print(f"Checked pattern: {mask_pattern}")

    print(f"Finished processing all MRI data")

def split_data(output_dir, train_ratio=0.8):
    print(f"Splitting data in {output_dir}")
    image_files = [f for f in os.listdir(f"{output_dir}/images") if f.endswith('.png')]
    train_files, val_files = train_test_split(image_files, train_size=train_ratio, random_state=42)

    os.makedirs(f"{output_dir}/train/images", exist_ok=True)
    os.makedirs(f"{output_dir}/train/labels", exist_ok=True)
    os.makedirs(f"{output_dir}/val/images", exist_ok=True)
    os.makedirs(f"{output_dir}/val/labels", exist_ok=True)

    for file in train_files:
        os.rename(f"{output_dir}/images/{file}", f"{output_dir}/train/images/{file}")
        os.rename(f"{output_dir}/labels/{file.replace('.png', '.txt')}", f"{output_dir}/train/labels/{file.replace('.png', '.txt')}")
        print(f"Moved {file} to train set")

    for file in val_files:
        os.rename(f"{output_dir}/images/{file}", f"{output_dir}/val/images/{file}")
        os.rename(f"{output_dir}/labels/{file.replace('.png', '.txt')}", f"{output_dir}/val/labels/{file.replace('.png', '.txt')}")
        print(f"Moved {file} to validation set")

    os.rmdir(f"{output_dir}/images")
    os.rmdir(f"{output_dir}/labels")
    print("Data split completed")

def create_data_yaml(output_dir):
    print(f"Creating data.yaml in {output_dir}")
    current_dir = os.getcwd()
    data = {
        'train': os.path.join(current_dir, output_dir, 'train', 'images'),
        'val': os.path.join(current_dir, output_dir, 'val', 'images'),
        'nc': 1,
        'names': ['airway']
    }

    with open(f"{output_dir}/data.yaml", 'w') as f:
        yaml.dump(data, f)
    print("data.yaml created")

def train_yolo_model(output_dir, epochs=400):
    print(f"Training YOLOv8 model with data from {output_dir}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load a model
    model = YOLO('yolov8m.yaml')  # build a new model from YAML
    
    data_yaml_path = os.path.join(os.getcwd(), output_dir, 'data.yaml')
    
    try:
        # Train the model
        results = model.train(data=data_yaml_path, epochs=epochs, imgsz=640, device=device)
        
        print("Training completed successfully")
        
        # The model is automatically saved after training
        print(f"Model saved at: {results.save_dir}")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
    
    return model

if __name__ == "__main__":
    image_dir = "./images"
    mask_dir = "./masks"
    output_dir = "yolo_data"

    print(f"Starting script execution")
    print(f"Working directory: {os.getcwd()}")
    print(f"Image directory: {image_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Output directory: {output_dir}")

    try:
        # Check if directories exist
        for dir_path in [image_dir, mask_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")

        # Process MRI data
        process_all_mri_data(image_dir, mask_dir, output_dir)

        # Check if output directories were created
        for subdir in ['train/images', 'train/labels', 'val/images', 'val/labels']:
            full_path = os.path.join(output_dir, subdir)
            if not os.path.exists(full_path):
                print(f"path {full_path} not exist")
                print("will create them")
                os.makedirs(full_path, exist_ok=True)
            else:
                print(f"Full path for yolo training: {full_path}")

        # Split data
        split_data(output_dir)

        # Create data.yaml
        create_data_yaml(output_dir)

        # Train model
        model = train_yolo_model(output_dir)

        print("Script execution completed successfully")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()