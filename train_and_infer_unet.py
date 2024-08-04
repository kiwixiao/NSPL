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

def train_unet(model, train_loader, val_loader, device, num_epochs=500, save_dir='unet_checkpoints'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    os.makefirs(save_dir, exist_ok=True)
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

def preprocess_mri_slice(slice_img, target_size=(256, 256)):
    # Normalize the slice
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
    # Resize the slice
    slice_img = zoom(slice_img, (target_size[0] / slice_img.shape[0], target_size[1] / slice_img.shape[1]))
    return slice_img

def infer_on_3d_mri(yolo_model, unet_model, mri_path, device):
    # Load the MRI
    mri_img = nib.load(mri_path)
    mri_data = mri_img.get_fdata()
    
    logging.info(f"Loaded MRI with shape: {mri_data.shape}")
    
    # Prepare the output segmentation mask
    seg_mask = np.zeros_like(mri_data)
    
    for i in range(mri_data.shape[2]):  # Iterate through slices
        slice_img = mri_data[:, :, i]
        processed_slice = preprocess_mri_slice(slice_img)
        
        # YOLO prediction
        yolo_results = yolo_model(processed_slice)
        
        if len(yolo_results[0].boxes) > 0:
            box = yolo_results[0].boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Crop the slice
            cropped_slice = processed_slice[int(y1):int(y2), int(x1):int(x2)]
            
            # Resize cropped slice to UNet input size
            cropped_slice = zoom(cropped_slice, (256 / cropped_slice.shape[0], 256 / cropped_slice.shape[1]))
            
            # Prepare for UNet
            cropped_tensor = torch.from_numpy(cropped_slice).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # UNet prediction
            with torch.no_grad():
                pred_mask = unet_model(cropped_tensor)
                pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
            
            # Resize mask back to cropped size
            pred_mask = zoom(pred_mask, (int(y2-y1) / 256, int(x2-x1) / 256))
            
            # Place the prediction in the full-size mask
            seg_mask[int(y1):int(y2), int(x1):int(x2), i] = pred_mask
        
        if i % 10 == 0:
            logging.info(f"Processed slice {i}/{mri_data.shape[2]}")
    
    # Save the segmentation mask
    nib.save(nib.Nifti1Image(seg_mask, mri_img.affine), 'predicted_segmentation.nii.gz')
    logging.info("Saved predicted segmentation mask")

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
        unet_model = UNet(n_channels=1, n_classes=1).to(device) # make sure the data pass to device. 
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
        
        # Load trained UNet model
        unet_model = UNet(n_channels=1, n_classes=1).to(device)
        unet_model.load_state_dict(torch.load('trained_unet.pth'))
        unet_model.eval()
        
        # Perform inference on a 3D MRI
        infer_on_3d_mri(yolo_model, unet_model, args.mri_path, device)
        
        logging.info("Inference completed. Check 'predicted_segmentation.nii.gz' for results.")