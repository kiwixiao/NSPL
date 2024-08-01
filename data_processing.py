# data_processing.py

import os
import logging
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import cv2
from PIL import Image

def load_and_preprocess_data(image_dir, mask_dir):
    logging.info("Loading and preprocessing data")
    images = []
    masks = []
    
    for img_file in os.listdir(image_dir):
        if img_file.endswith('.nii') or img_file.endswith('.nii.gz'):
            img_path = os.path.join(image_dir, img_file)
            subjectID = img_file.split('_')[0] # this should give me the sbjectID OSAMRI001 etc.
            mask_file = subjectID+"*_mask.nii*" # this should give me OSAMRI*nii*
            mask_path = os.path.join(mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                img = nib.load(img_path).get_fdata()
                mask = nib.load(mask_path).get_fdata()
                
                # Resample to 256x256x256 with 1.0 isotropic spacing
                img_resampled = resample_to_isotropic(img)
                mask_resampled = resample_to_isotropic(mask)
                
                images.append(img_resampled)
                masks.append(mask_resampled)
                logging.info(f"Loaded and preprocessed {img_file}")
            else:
                logging.warning(f"Mask not found for {img_file}")
    
    return np.array(images), np.array(masks)

def resample_to_isotropic(volume, target_shape=(256, 256, 256)):
    current_shape = volume.shape
    scale_factors = [t / c for t, c in zip(target_shape, current_shape)]
    return zoom(volume, scale_factors, order=1)

def save_image(image, filepath):
    """
    Save the image slice as a PNG file.
    
    :param image: 2D numpy array representing the image slice
    :param filepath: Path to save the image file
    """
    # Normalize the image to 0-255 range
    normalized_image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # Create a PIL Image and save it
    img = Image.fromarray(normalized_image)
    img.save(filepath)
    logging.info(f"Saved image: {filepath}")

def save_yolo_labels(bboxes, image_shape, filepath):
    """
    Save the bounding boxes in YOLO format.
    
    :param bboxes: List of bounding boxes [x, y, w, h]
    :param image_shape: Shape of the original image (height, width)
    :param filepath: Path to save the label file
    """
    height, width = image_shape
    with open(filepath, 'w') as f:
        for bbox in bboxes:
            x, y, w, h = bbox
            # Convert to YOLO format: class x_center y_center width height
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w = w / width
            h = h / height
            
            # Assuming class 0 for all bounding boxes (modify if needed)
            f.write(f"0 {x_center} {y_center} {w} {h}\n")
    
    logging.info(f"Saved YOLO labels: {filepath}")

def create_bounding_boxes(mask, margin=1.2):
    """
    Create bounding boxes from a binary mask.
    
    :param mask: Binary mask
    :param margin: Margin to add around the bounding box
    :return: List of bounding boxes [x, y, w, h]
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Apply margin
        center_x, center_y = x + w/2, y + h/2
        w *= margin
        h *= margin
        x = max(0, int(center_x - w/2))
        y = max(0, int(center_y - h/2))
        w = min(mask.shape[1] - x, int(w))
        h = min(mask.shape[0] - y, int(h))
        
        bboxes.append([x, y, w, h])
    
    return bboxes

def prepare_data_for_yolo(images, masks, output_dir):
    logging.info("Preparing data for YOLOv8")
    yolo_data = []
    
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    for i, (image, mask) in enumerate(zip(images, masks)):
        for slice_idx in range(image.shape[2]):
            img_slice = image[:, :, slice_idx]
            mask_slice = mask[:, :, slice_idx]
            
            if np.any(mask_slice):  # Only process slices with labels
                bboxes = create_bounding_boxes(mask_slice)
                
                if bboxes:
                    img_filename = f"slice_{i}_{slice_idx}.png"
                    label_filename = f"slice_{i}_{slice_idx}.txt"
                    
                    # Save image
                    save_image(img_slice, os.path.join(output_dir, "images", img_filename))
                    
                    # Save labels
                    save_yolo_labels(bboxes, mask_slice.shape, os.path.join(output_dir, "labels", label_filename))
                    
                    yolo_data.append((img_filename, label_filename))
    
    logging.info(f"Prepared {len(yolo_data)} slices for YOLOv8")
    return yolo_data