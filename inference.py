import os
import argparse
import numpy as np
import nibabel as nib
import torch
from models import UNetResNet, nnUNet, SimpleUNet
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

def normalize_image(image):
    centered = (image - image.mean()) / image.std()
    normalized = (centered - centered.min()) / (centered.max() - centered.min())
    return normalized

def resample_volume(volume, target_spacing=(0.6, 0.6, 0.6)):
    current_spacing = volume.header.get_zooms()[:3]
    current_shape = volume.shape[:3]
    scale_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
    new_shape = np.round(current_shape * np.array(scale_factors)).astype(int)
    resampled = zoom(volume.get_fdata(), scale_factors, order=3, mode='constant')
    achieved_spacing = [c * s / n for c, s, n in zip(current_spacing, current_shape, new_shape)]
    return resampled, achieved_spacing

def prepare_for_yolo(slice_img):
    yolo_input = np.stack([slice_img] * 3, axis=-1)
    yolo_input = cv2.resize(yolo_input, (640, 640))
    yolo_input = np.transpose(yolo_input, (2, 0, 1)).astype(np.float32)
    yolo_input = np.expand_dims(yolo_input, 0)
    return torch.from_numpy(yolo_input)

def crop_image(image, bbox):
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]

def infer(model, yolo_model, input_path, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input MRI
    mri_img = nib.load(input_path)
    original_data = mri_img.get_fdata()
    original_shape = original_data.shape
    original_affine = mri_img.affine
    original_header = mri_img.header
    
    # Preprocess
    mri_canonical = nib.as_closest_canonical(mri_img)
    mri_resampled, achieved_spacing = resample_volume(mri_canonical)
    mri_normalized = normalize_image(mri_resampled)
    
    # Prepare for segmentation
    segmentation = np.zeros_like(mri_normalized)
    
    for i in range(mri_normalized.shape[1]):  # Coronal slices
        slice_img = mri_normalized[:, i, :]
        
        # YOLO detection
        yolo_input = prepare_for_yolo(slice_img)
        yolo_results = yolo_model(yolo_input)
        
        # Process YOLO results
        if len(yolo_results[0].boxes) > 0:
            bbox = yolo_results[0].boxes[0].xyxy[0].cpu().numpy()
            cropped_img = crop_image(slice_img, bbox)
            
            # Resize for UNet
            cropped_resized = cv2.resize(cropped_img, (256, 256))
            
            # UNet prediction
            with torch.no_grad():
                input_tensor = torch.from_numpy(cropped_resized).float().unsqueeze(0).unsqueeze(0).to(device)
                output = model(input_tensor)
                pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # Resize prediction back to cropped size
            pred_mask_resized = cv2.resize(pred_mask, (int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])))
            
            # Place prediction in full-size mask
            segmentation[int(bbox[1]):int(bbox[3]), i, int(bbox[0]):int(bbox[2])] = pred_mask_resized
            
            # Save intermediate results
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cropped_img, cmap='gray')
            plt.title('Cropped Input')
            plt.subplot(1, 2, 2)
            plt.imshow(pred_mask, cmap='jet')
            plt.title('Predicted Mask')
            plt.savefig(os.path.join(output_dir, f'slice_{i:03d}_results.png'))
            plt.close()
    
    # Transform segmentation back to original space
    segmentation_original_space = nib.Nifti1Image(segmentation, mri_canonical.affine)
    segmentation_original_space = nib.as_closest_canonical(segmentation_original_space)
    
    # Resample segmentation to match original image exactly
    segmentation_resampled = nib.processing.resample_to_img(segmentation_original_space, mri_img, order=0)
    
    # Ensure the header matches the original
    segmentation_resampled.header.set_zooms(mri_img.header.get_zooms())
    
    # Save the final segmentation
    output_path = os.path.join(output_dir, 'predicted_segmentation.nii.gz')
    nib.save(segmentation_resampled, output_path)
    print(f"Saved predicted segmentation to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Infer segmentation on new 3D MRI NIFTI")
    parser.add_argument('--input_path', type=str, required=True, help="Path to input 3D MRI NIFTI")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save results")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained UNet model")
    parser.add_argument('--yolo_model_path', type=str, required=True, help="Path to trained YOLO model")
    parser.add_argument('--model_type', choices=['simple_unet', 'nnunet', 'unet_resnet'], required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load UNet model
    if args.model_type == 'simple_unet':
        model = SimpleUNet(n_channels=1, n_classes=1)
    elif args.model_type == 'nnunet':
        model = nnUNet(n_channels=1, n_classes=1)
    else:
        model = UNetResNet(n_channels=1, n_classes=1)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load YOLO model
    yolo_model = YOLO(args.yolo_model_path)

    infer(model, yolo_model, args.input_path, args.output_dir, device)

if __name__ == "__main__":
    
    """
    python inference.py 
    --input_path /path/to/input.nii.gz 
    --output_dir /path/to/output 
    --model_path /path/to/unet_model.pth 
    --yolo_model_path /path/to/yolo_model.pt 
    --model_type unet_resnet
    """
    
    main()