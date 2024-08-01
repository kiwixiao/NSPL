# unet_infer.py

import os
import logging
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from scipy.ndimage import zoom
from ultralytics import YOLO
from unet_segmentation import UNet  # Import the UNet model from your training script

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

def load_yolo_model(model_path=None):
    if model_path and os.path.exists(model_path):
        return YOLO(model_path)
    
    # Find the latest trained YOLO model
    run_dirs = sorted(glob.glob('runs/detect/train*'), key=os.path.getmtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError("No trained YOLO model found.")
    
    model_path = os.path.join(run_dirs[0], 'weights', 'best.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No 'best.pt' file found in {run_dirs[0]}")
    
    logging.info(f"Loading YOLO model from: {model_path}")
    return YOLO(model_path)

def load_unet_model(model_path=None):
    if model_path is None:
        model_path = 'unet_model_best.pth'  # Default to the best model
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"U-Net model not found at {model_path}")
    
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logging.info(f"Loaded U-Net model from {model_path}")
    return model

def resample_to_isotropic(volume, target_shape=(256, 256, 256)):
    current_shape = volume.shape
    scale_factors = [t / c for t, c in zip(target_shape, current_shape)]
    return zoom(volume, scale_factors, order=1)

def resample_to_original(volume, original_shape):
    return zoom(volume, [o / c for o, c in zip(original_shape, volume.shape)], order=1)

def crop_image(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]

def process_slice(slice_img, yolo_model, unet_model):
    # Normalize the slice to 0-255 range for YOLO
    slice_img_norm = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
    
    # YOLO inference
    result = yolo_model(slice_img_norm)
    
    if len(result) > 0 and len(result[0].boxes) > 0:
        box = result[0].boxes[0]  # Take the first box (assuming single airway per slice)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Crop the image
        cropped_img = crop_image(slice_img, (x1, y1, x2, y2))
        
        # Prepare for U-Net
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        cropped_tensor = transform(cropped_img).unsqueeze(0).to(device)
        
        # U-Net inference
        with torch.no_grad():
            output = unet_model(cropped_tensor)
            pred = torch.sigmoid(output) > 0.5
        
        # Convert prediction back to numpy and original size
        pred_np = pred.squeeze().cpu().numpy().astype(float)
        full_pred = np.zeros_like(slice_img)
        full_pred[y1:y2, x1:x2] = zoom(pred_np, (y2-y1)/(pred_np.shape[0]), (x2-x1)/(pred_np.shape[1]))
        
        return full_pred
    else:
        return np.zeros_like(slice_img)

def infer_3d_image(image_path, yolo_model, unet_model):
    # Load the image
    img_nifti = nib.load(image_path)
    original_img = img_nifti.get_fdata()
    original_shape = original_img.shape
    
    # Resample to isotropic
    img_isotropic = resample_to_isotropic(original_img)
    
    # Process each slice
    predictions = []
    for i in range(img_isotropic.shape[2]):
        slice_img = img_isotropic[:, :, i]
        pred_slice = process_slice(slice_img, yolo_model, unet_model)
        predictions.append(pred_slice)
    
    # Stack predictions
    pred_volume = np.stack(predictions, axis=2)
    
    # Resample back to original shape
    final_pred = resample_to_original(pred_volume, original_shape)
    
    return final_pred, img_nifti.affine

def main():
    # Load models
    yolo_model = load_yolo_model()
    unet_model = load_unet_model()  # This will load the best model by default
    
    # Get input image path
    image_path = input("Enter the path to the 3D image for inference: ")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The specified image path does not exist: {image_path}")
    
    # Perform inference
    logging.info(f"Processing image: {image_path}")
    pred_mask, affine = infer_3d_image(image_path, yolo_model, unet_model)
    
    # Save the result
    output_path = os.path.join(os.path.dirname(image_path), f"{os.path.basename(image_path).split('.')[0]}_segmentation.nii.gz")
    nib.save(nib.Nifti1Image(pred_mask, affine), output_path)
    logging.info(f"Segmentation mask saved to: {output_path}")

if __name__ == "__main__":
    main()