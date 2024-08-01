# yolo_inference.py

import os
import logging
import glob
import nibabel as nib
import numpy as np
from ultralytics import YOLO
import cv2
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def resample_to_isotropic(volume, target_shape=(256, 256, 256)):
    current_shape = volume.shape
    scale_factors = [t / c for t, c in zip(target_shape, current_shape)]
    return zoom(volume, scale_factors, order=1)

def load_latest_yolo_model(model_path=None):
    if model_path:
        if os.path.exists(model_path):
            logging.info(f"Loading YOLO model from provided path: {model_path}")
            return YOLO(model_path)
        else:
            raise FileNotFoundError(f"The provided model path does not exist: {model_path}")
    
    # If no model_path is provided, find the latest run directory
    run_dirs = glob.glob('runs/detect/train*')
    if not run_dirs:
        raise FileNotFoundError("No trained YOLO model found. Please train a model first.")
    
    latest_run = max(run_dirs, key=os.path.getmtime)
    
    # Find the best.pt file in the latest run directory
    model_path = os.path.join(latest_run, 'weights', 'best.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No 'best.pt' file found in {latest_run}")
    
    logging.info(f"Loading latest YOLO model from: {model_path}")
    return YOLO(model_path)

def crop_image(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]

def plot_cropped_slices(images, masks, num_slices=5):
    num_images = len(images)
    indices = np.linspace(0, num_images - 1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(num_slices, 2, figsize=(10, 4 * num_slices))
    fig.suptitle("Cropped Images and Masks", fontsize=16)
    
    for i, idx in enumerate(indices):
        # Plot image
        axes[i, 0].imshow(images[idx], cmap='gray')
        axes[i, 0].set_title(f"Image Slice {idx}")
        axes[i, 0].axis('off')
        
        # Plot mask
        axes[i, 1].imshow(masks[idx], cmap='gray')
        axes[i, 1].set_title(f"Mask Slice {idx}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def process_single_image(image_path, mask_path, model, output_dir):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The specified image path does not exist: {image_path}")
    
    # Load and preprocess the image
    logging.info(f"Loading and preprocessing image: {image_path}")
    img = nib.load(image_path).get_fdata()
    img_resampled = resample_to_isotropic(img)
    
    # Load and preprocess the mask if available
    if mask_path and os.path.exists(mask_path):
        logging.info(f"Loading and preprocessing mask: {mask_path}")
        mask = nib.load(mask_path).get_fdata()
        mask_resampled = resample_to_isotropic(mask)
    else:
        mask_resampled = None
    
    # Perform inference and crop images/masks
    logging.info(f"Performing inference and cropping images/masks")
    cropped_images = []
    cropped_masks = []
    for slice_idx in range(img_resampled.shape[2]):
        slice_img = img_resampled[:, :, slice_idx]
        # Normalize the slice to 0-255 range
        slice_img_norm = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
        result = model(slice_img_norm)
        
        if len(result) > 0 and len(result[0].boxes) > 0:  # Check if any detections in this slice
            box = result[0].boxes[0]  # Take the first box (assuming single airway per slice)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Crop and save image
            cropped_img = crop_image(slice_img, (x1, y1, x2, y2))
            np.save(os.path.join(output_dir, "cropped_iso_images", f"{os.path.basename(image_path)}_slice_{slice_idx}.npy"), cropped_img)
            cropped_images.append(cropped_img)
            
            # Crop and save mask if available
            if mask_resampled is not None:
                slice_mask = mask_resampled[:, :, slice_idx]
                cropped_mask = crop_image(slice_mask, (x1, y1, x2, y2))
                np.save(os.path.join(output_dir, "cropped_iso_masks", f"{os.path.basename(mask_path)}_slice_{slice_idx}.npy"), cropped_mask)
                cropped_masks.append(cropped_mask)
            
            logging.info(f"Processed and saved cropped slice {slice_idx}")
    
    return cropped_images, cropped_masks

def perform_yolo_inference(model_path=None):
    # Load the model (either from provided path or latest trained)
    model = load_latest_yolo_model(model_path)
    
    # Prompt for image and mask directories
    image_dir = input("Enter the path to the directory containing image(s) for inference: ")
    mask_dir = input("Enter the path to the directory containing corresponding mask(s) (or press Enter if not available): ")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"The specified image directory does not exist: {image_dir}")
    
    # Create output directories
    output_dir = "yolo_inference_output"
    os.makedirs(os.path.join(output_dir, "cropped_iso_images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cropped_iso_masks"), exist_ok=True)
    
    # Get list of image files
    image_files = glob.glob(os.path.join(image_dir, "*.nii*"))
    
    if not image_files:
        raise FileNotFoundError(f"No NIfTI images found in the directory: {image_dir}")
    
    for image_path in image_files:
        logging.info(f"Processing image: {image_path}")
        
        # Find corresponding mask if mask directory is provided
        mask_path = None
        if mask_dir:
            mask_name = os.path.basename(image_path).replace("_mri", "_mask")
            mask_path = os.path.join(mask_dir, mask_name)
            if not os.path.exists(mask_path):
                logging.warning(f"Corresponding mask not found for {image_path}")
                mask_path = None
        
        # Process the image
        cropped_images, cropped_masks = process_single_image(image_path, mask_path, model, output_dir)
        
        # Plot cropped slices for visual inspection
        if cropped_images and cropped_masks:
            plot_cropped_slices(cropped_images, cropped_masks)
        elif cropped_images:
            plot_cropped_slices(cropped_images, cropped_images)  # If no masks, show images twice
        else:
            logging.warning(f"No cropped slices to display for {image_path}")
    
    logging.info("Inference and cropping completed for all images")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        perform_yolo_inference()
    except Exception as e:
        logging.error(f"An error occurred during inference and cropping: {str(e)}", exc_info=True)