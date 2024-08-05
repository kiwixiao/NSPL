# train_and_infer_unet.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from unet_model import UNet, check_input_dimension
from ultralytics import YOLO
import nibabel as nib
from scipy.ndimage import zoom
import argparse
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# Dataset class for loading airway images and masks
class AirwayDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        # Resize images to a fixed size (e.g., 256x256)
        image = image.resize((256, 256), Image.BILINEAR)
        mask = mask.resize((256, 256), Image.NEAREST)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image.to(device), mask.to(device)

# Training function for UNet model
def train_unet(model, train_loader, val_loader, device, num_epochs=500, save_dir='unet_checkpoints'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # Use tqdm for a progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{train_loss/train_batches:.4f}'})
        
        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Log the losses
        logging.info(f'Epoch {epoch+1}/{num_epochs}, '
                     f'Train Loss: {avg_train_loss:.4f}, '
                     f'Validation Loss: {avg_val_loss:.4f}')
        # save the model every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"unet_epoch_{epoch+1}.pth"))
            logging.info(f'Model saved at epoch {epoch+1}')
            
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_unet_model.pth'))
            logging.info(f'New best model saved with validation loss: {best_val_loss:.4f}')
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_unet_model.pth'))
    logging.info("Training completed. Final model saved.")
    
    # Plot and save training metrics
    plot_training_metrics(train_losses, val_losses, save_dir)
    
    return model, train_losses, val_losses

def plot_training_metrics(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, 'train_loss_plot.png'))
    plt.close()

    # Plot validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(val_losses)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, 'val_loss_plot.png'))
    plt.close()



def preprocess_mri_slice(slice_img):
    # Normalize the slice
    slice_img = normalize_image(slice_img)
    return slice_img

def save_intermediate_result(img, filename):
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def infer_on_3d_mri(yolo_model, unet_model, mri_path, device):
    output_folder = os.path.dirname(mri_path)
    base_filename = os.path.basename(mri_path).rsplit('.', 1)[0]
    yolo_debug_folder = os.path.join(output_folder, f'{base_filename}_yolo_debug')
    overlay_folder = os.path.join(output_folder, f'{base_filename}_unet_overlay')
    cropped_folder = os.path.join(output_folder, f'{base_filename}_cropped_images')
    os.makedirs(yolo_debug_folder, exist_ok=True)
    os.makedirs(overlay_folder, exist_ok=True)
    os.makedirs(cropped_folder, exist_ok=True)

    # Load MRI and get original affine and shape
    mri_img = nib.load(mri_path)
    original_affine = mri_img.affine
    original_shape = mri_img.shape
    mri_data = mri_img.get_fdata()

    # Reorient to canonical orientation
    mri_canonical = nib.as_closest_canonical(mri_img)
    canonical_affine = mri_canonical.affine
    mri_data_canonical = mri_canonical.get_fdata()

    # Resample and normalize MRI
    mri_resampled, _ = resample_volume(mri_canonical)
    mri_normalized = normalize_image(mri_resampled)

    seg_mask = np.zeros_like(mri_normalized)

    for i in range(mri_normalized.shape[1]):  # Coronal slices
        slice_img = mri_normalized[:, i, :]
        original_slice_shape = slice_img.shape

        # Prepare for YOLO
        yolo_input = prepare_for_yolo(slice_img)
        
        # YOLO prediction
        yolo_results = yolo_model(yolo_input)

        # Save YOLO prediction with bounding box
        yolo_debug_img = (slice_img * 255).astype(np.uint8)
        yolo_debug_img = cv2.cvtColor(yolo_debug_img, cv2.COLOR_GRAY2BGR)

        all_boxes = []
        for r in yolo_results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(yolo_debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if all_boxes:
            all_boxes = np.array(all_boxes)
            x1, y1 = all_boxes[:, :2].min(axis=0)
            x2, y2 = all_boxes[:, 2:].max(axis=0)

            # Rescale bounding box to original dimensions
            x1, y1 = int(x1 * original_slice_shape[1] / 640), int(y1 * original_slice_shape[0] / 640)
            x2, y2 = int(x2 * original_slice_shape[1] / 640), int(y2 * original_slice_shape[0] / 640)

            # Ensure the cropped area is valid
            x1, x2 = max(0, x1), min(original_slice_shape[1], x2)
            y1, y2 = max(0, y1), min(original_slice_shape[0], y2)

            if x1 >= x2 or y1 >= y2:
                logging.warning(f"Invalid bounding box for slice {i}. Skipping.")
                continue

            # Draw combined bounding box
            cv2.rectangle(yolo_debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Crop
            cropped_slice = slice_img[y1:y2, x1:x2]

            # Resize for UNet while maintaining aspect ratio
            unet_input = resize_and_pad(cropped_slice, (256, 256))

            # Save cropped image for debugging
            cropped_filename = os.path.join(cropped_folder, f'{base_filename}_slice_{i:03d}_crop.png')
            cv2.imwrite(cropped_filename, (cropped_slice * 255).astype(np.uint8))

            # UNet prediction
            with torch.no_grad():
                unet_input_tensor = torch.from_numpy(unet_input).float().unsqueeze(0).unsqueeze(0).to(device)
                pred_mask = unet_model(unet_input_tensor)
                pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()

            # Resize mask to match the exact dimensions of the cropped area
            pred_mask_resized = cv2.resize(pred_mask, (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR)

            # Threshold the mask
            pred_mask_resized = (pred_mask_resized > 0.5).astype(np.float32)

            # Place in full-size mask
            seg_mask[y1:y2, i, x1:x2] = pred_mask_resized

            # Save overlay images
            save_overlay_images(cropped_slice, pred_mask_resized, i, overlay_folder)

        # Save YOLO debug image
        cv2.imwrite(os.path.join(yolo_debug_folder, f'{base_filename}_yolo_debug_slice_{i:03d}.png'), yolo_debug_img)

        if i % 10 == 0:
            logging.info(f"Processed slice {i}/{mri_normalized.shape[1]}")

    # Resize segmentation mask back to original MRI dimensions
    seg_mask_original = resize_to_original(seg_mask, original_shape)

    # Threshold the resized mask
    seg_mask_original = (seg_mask_original > 0.5).astype(np.float32)

    # Reorient the segmentation mask to match the original orientation
    seg_mask_img = nib.Nifti1Image(seg_mask_original, canonical_affine)
    seg_mask_img = nib.as_closest_canonical(seg_mask_img)
    seg_mask_img = nib.Nifti1Image(seg_mask_img.get_fdata(), original_affine)

    # Save the full segmentation mask in original dimensions and orientation
    output_filename = os.path.basename(mri_path).rsplit('.', 1)[0] + '_pred_seg.nii.gz'
    output_path = os.path.join(output_folder, output_filename)
    nib.save(seg_mask_img, output_path)
    logging.info(f"Saved predicted segmentation mask: {output_path}")

    # Save a debug version of the mask
    debug_mask = (seg_mask_original * 255).astype(np.uint8)
    debug_mask_img = nib.Nifti1Image(debug_mask, original_affine)
    debug_output_path = os.path.join(output_folder, f"{base_filename}_pred_seg_debug.nii.gz")
    nib.save(debug_mask_img, debug_output_path)
    logging.info(f"Saved debug segmentation mask: {debug_output_path}")

def resize_to_original(mask, original_shape):
    return zoom(mask, (original_shape[0] / mask.shape[0],
                       original_shape[1] / mask.shape[1],
                       original_shape[2] / mask.shape[2]), order=1)

def resize_and_pad(image, target_size):
    h, w = image.shape
    target_h, target_w = target_size
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    padded = np.pad(resized, ((pad_h, target_h - new_h - pad_h), (pad_w, target_w - new_w - pad_w)), mode='constant')
    return padded

def prepare_for_yolo(slice_img):
    yolo_input = np.stack([slice_img] * 3, axis=-1)  # Convert to 3 channels
    yolo_input = cv2.resize(yolo_input, (640, 640))  # Resize to YOLO input size
    yolo_input = np.transpose(yolo_input, (2, 0, 1)).astype(np.float32)  # HWC to CHW
    yolo_input = np.expand_dims(yolo_input, 0)  # Add batch dimension
    return torch.from_numpy(yolo_input)

def resample_volume(volume, target_spacing=(0.6, 0.6, 0.6), is_mask=False):
    current_spacing = volume.header.get_zooms()[:3]
    current_shape = volume.shape[:3]
    scale_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
    new_shape = np.round(current_shape * np.array(scale_factors)).astype(int)
    if is_mask:
        resampled = zoom(volume.get_fdata(), scale_factors, order=0, mode='nearest')
        resampled = (resampled > 0.5).astype(np.uint8)
    else:
        resampled = zoom(volume.get_fdata(), scale_factors, order=3, mode='constant')
    return resampled, [c * s / n for c, s, n in zip(current_spacing, current_shape, new_shape)]

def normalize_image(image):
    centered = (image - image.mean()) / image.std()
    normalized = (centered - centered.min()) / (centered.max() - centered.min())
    return normalized

def save_overlay_images(cropped_slice, pred_mask, slice_idx, output_folder):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot cropped image
    ax1.imshow(cropped_slice, cmap='gray')
    ax1.set_title('Cropped MRI Slice')
    ax1.axis('off')
    
    # Plot predicted mask
    ax2.imshow(pred_mask, cmap='jet', alpha=0.7)
    ax2.set_title('Predicted Mask')
    ax2.axis('off')
    
    # Plot overlay
    ax3.imshow(cropped_slice, cmap='gray')
    ax3.imshow(pred_mask, cmap='jet', alpha=0.3)
    ax3.set_title('Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'overlay_slice_{slice_idx:03d}.png'))
    plt.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Train UNet or perform inference on 3D MRI")
    parser.add_argument('--mode', type=str, choices=['train', 'infer'], required=True, help='Mode: train or infer')
    parser.add_argument('--mri_path', type=str, help='Path to 3D MRI for inference', required=False)
    args = parser.parse_args()

    # Set up paths
    seg_data_dir = "yolo_data/seg_data"
    image_dir = os.path.join(seg_data_dir, "images")
    mask_dir = os.path.join(seg_data_dir, "masks")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    if args.mode == 'train':
        # Create dataset and dataloaders
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        dataset = AirwayDataset(image_dir, mask_dir, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Create and train UNet model
        unet_model = UNet(n_channels=1, n_classes=1).to(device)
        check_input_dimension(unet_model)
        
        # Log the start of training
        logging.info("Starting UNet training...")
        start_time = time.time()
        
        trained_model, train_losses, val_losses = train_unet(unet_model, train_loader, 
                                                             val_loader, device, num_epochs=500, save_dir='unet_checkpoints')

        # Log training completion and duration
        end_time = time.time()
        training_duration = end_time - start_time
        logging.info(f"Training completed in {training_duration:.2f} seconds.")
        
        logging.info("Training completed and model saved")
    
    elif args.mode == 'infer':
        if args.mri_path is None:
            raise ValueError("MRI path is required for inference mode")
        
        # Load YOLO model
        yolo_model = YOLO('./runs/detect/train3/weights/best.pt')
        unet_model = UNet(n_channels=1, n_classes=1).to(device)
        unet_model.load_state_dict(torch.load('./unet_checkpoints/best_unet_model.pth', map_location=device))
        unet_model.eval()

        # Perform inference on the 3D MRI
        infer_on_3d_mri(yolo_model, unet_model, args.mri_path, device)
    
        logging.info(f"Inference completed. Check {os.path.dirname(args.mri_path)} for results.")