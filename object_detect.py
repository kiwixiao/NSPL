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
import argparse

def find_objects(mask, margin=1.2):
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    # Combine all contours
    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Calculate center
    center_x = x + w / 2
    center_y = y + h / 2
    
    # Apply margin
    w_with_margin = min(w * margin, mask.shape[1])
    h_with_margin = min(h * margin, mask.shape[0])
    
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
    
    # Only return if the object is not covering the entire image
    if width < 0.9 and height < 0.9:
        return [0, center_x, center_y, width, height]  # 0 is the class index for airway
    else:
        return None
    
def plot_image_with_box(image, box, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    if box is not None:
        _, center_x, center_y, width, height = box
        x = (center_x - width/2) * image.shape[1]
        y = (center_y - height/2) * image.shape[0]
        rect = plt.Rectangle((x, y), width * image.shape[1], height * image.shape[0],
                             fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def save_cropped_data(image, mask, box, image_output_path, mask_output_path):
    if box is not None:
        _, center_x, center_y, width, height = box
        x = int((center_x - width/2) * image.shape[1])
        y = int((center_y - height/2) * image.shape[0])
        w = int(width * image.shape[1])
        h = int(height * image.shape[0])
        
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        
        Image.fromarray((cropped_image * 255).astype(np.uint8)).save(image_output_path)
        Image.fromarray((cropped_mask * 255).astype(np.uint8)).save(mask_output_path)

def save_image(img, filename):
    # Rescale to [0, 255] for saving as PNG
    img_rescaled = (img * 255).astype(np.uint8)
    Image.fromarray(img_rescaled, mode='L').save(filename)
    print(f"Saved image: {filename}")

def save_yolo_annotation(objects, filename):
    with open(filename, 'w') as f:
        for obj in objects:
            f.write(' '.join(map(str, obj)) + '\n')
    print(f"Saved annotation: {filename}")

def resample_volume(volume, target_spacing=(0.6, 0.6, 0.6), is_mask=False):
    # Get current spacing and shape
    current_spacing = volume.header.get_zooms()[:3]
    current_shape = volume.shape[:3]

    # Calculate the scaling factors
    scale_factors = [c / t for c, t in zip(current_spacing, target_spacing)]

    # Calculate new shape to maintain FOV
    new_shape = np.round(current_shape * np.array(scale_factors)).astype(int)

    # Resample
    if is_mask:
        resampled = zoom(volume.get_fdata(), scale_factors, order=0, mode='nearest')
        resampled = (resampled > 0.5).astype(np.uint8)
    else:
        resampled = zoom(volume.get_fdata(), scale_factors, order=3, mode='constant')

    # Calculate the actual achieved spacing
    achieved_spacing = [c * s / n for c, s, n in zip(current_spacing, current_shape, new_shape)]

    return resampled, achieved_spacing

def normalize_image(image):
    
    # Center around 0 with standard deviation 1
    centered = (image - image.mean()) / image.std()
    # Normalize to [0, 1]
    normalized = (centered - centered.min()) / (centered.max() - centered.min())
    
    return normalized

def save_nifti(data, original_img, new_spacing, filename):
    # Create a new affine matrix with the new spacing
    new_affine = original_img.affine.copy()
    np.fill_diagonal(new_affine[:3, :3], new_spacing)

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
    new_img.set_qform(new_affine)
    new_img.set_sform(new_affine)
    
    # Save the image
    nib.save(new_img, filename)
    print(f"Saved NIfTI file: {filename}")

def process_mri_to_yolo(mri_path, mask_path, output_dir, subject_id):
    print(f"Processing MRI: {mri_path}")
    print(f"Processing Mask: {mask_path}")
    
    # Load and reorient MRI and mask
    mri_img = nib.load(mri_path)
    mask_img = nib.load(mask_path)
    
    mri_img = nib.as_closest_canonical(mri_img)
    mask_img = nib.as_closest_canonical(mask_img)
    
    # Resample to target spacing while maintaining FOV
    mri_resampled, mri_spacing = resample_volume(mri_img)
    mask_resampled, mask_spacing = resample_volume(mask_img, is_mask=True)
    
    print(f"Original MRI shape: {mri_img.shape}, Resampled MRI shape: {mri_resampled.shape}")
    print(f"Original Mask shape: {mask_img.shape}, Resampled Mask shape: {mask_resampled.shape}")
    print(f"New spacing: {mri_spacing}")
    
    # Normalize and center the resampled MRI
    mri_normalized = normalize_image(mri_resampled)
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images_with_box"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg_data", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg_data", "masks"), exist_ok=True)
    
    # Process coronal slices
    for i in range(mri_normalized.shape[1]):  # Coronal slices
        slice_img = mri_normalized[:,i,:]
        slice_mask = mask_resampled[:,i,:]
        
        box = find_objects(slice_mask)
        
        if box is not None:
            img_filename = f"{output_dir}/images/{subject_id}_{i:03d}.png"
            save_image(slice_img, img_filename)
            
            txt_filename = f"{output_dir}/labels/{subject_id}_{i:03d}.txt"
            save_yolo_annotation([box], txt_filename)
            
            box_img_filename = f"{output_dir}/images_with_box/{subject_id}_{i:03d}.png"
            plot_image_with_box(slice_img, box, box_img_filename)
            
            seg_img_filename = f"{output_dir}/seg_data/images/{subject_id}_{i:03d}.png"
            seg_mask_filename = f"{output_dir}/seg_data/masks/{subject_id}_{i:03d}.png"
            save_cropped_data(slice_img, slice_mask, box, seg_img_filename, seg_mask_filename)
        else:
            print(f"No valid objects found in coronal slice {i} of {subject_id}")
    
    print(f"Processed all coronal slices for {subject_id}")

def process_all_mri_data(image_dir, mask_dir, output_dir):
    print(f"Processing all MRI data:")
    print(f"Image directory: {image_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Output directory: {output_dir}")

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    os.makedirs(f"{output_dir}/resampled/images", exist_ok=True)
    os.makedirs(f"{output_dir}/resampled/masks", exist_ok=True)

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
        'names': ['airway'],
        'ch': 1 # number of channeld, 1 ffor gray
    }

    with open(f"{output_dir}/data.yaml", 'w') as f:
        yaml.dump(data, f)
    print("data.yaml created")

def train_yolo_model(data_yaml_path, epochs=500):
    print(f"Training YOLOv8 model with data from {data_yaml_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load a model
    model = YOLO('yolov8m.yaml')  # build a new model from YAML
    
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
    parser = argparse.ArgumentParser(description="Process MRI data and/or train YOLO model")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip preprocessing and use existing data")
    parser.add_argument("--yolo_data_path", type=str, help="Path to existing YOLO data directory")
    parser.add_argument("--skip_training", action="store_true", help="Skip YOLO model training")
    args = parser.parse_args()

    if args.skip_preprocessing and args.yolo_data_path:
        # Use existing preprocessed data
        output_dir = args.yolo_data_path
        data_yaml_path = os.path.join(output_dir, 'data.yaml')
        
        if not os.path.exists(data_yaml_path):
            print(f"Error: data.yaml not found in {output_dir}")
            exit(1)
        
        print(f"Using existing preprocessed data from: {output_dir}")
    else:
        # Perform preprocessing
        image_dir = "./images"
        mask_dir = "./masks"
        output_dir = "yolo_data"
        os.makedirs(output_dir, exist_ok=True)

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
            for subdir in ['images', 'labels', 'images_with_box', 'seg_data/images', 'seg_data/masks']:
                full_path = os.path.join(output_dir, subdir)
                if not os.path.exists(full_path):
                    print(f"Path {full_path} does not exist")
                    print("Will create it")
                    os.makedirs(full_path, exist_ok=True)
                else:
                    print(f"Full path created: {full_path}")

            # Split data
            split_data(output_dir)

            # Create data.yaml
            create_data_yaml(output_dir)

            data_yaml_path = os.path.join(output_dir, 'data.yaml')

            print("Preprocessing completed successfully")

        except Exception as e:
            print(f"An error occurred during preprocessing: {str(e)}")
            import traceback
            traceback.print_exc()
            exit(1)

    if not args.skip_training:
        # Train model
        try:
            model = train_yolo_model(data_yaml_path)
            print("Model training completed")
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping model training as per user request")

    print("Script execution completed successfully")