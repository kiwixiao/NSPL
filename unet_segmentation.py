# unet_segmentation.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.encoder = nn.ModuleList([
            double_conv(in_channels, 64),
            double_conv(64, 128),
            double_conv(128, 256),
            double_conv(256, 512),
        ])
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.middle = double_conv(512, 1024)
        
        self.decoder = nn.ModuleList([
            double_conv(1024, 512),
            double_conv(512, 256),
            double_conv(256, 128),
            double_conv(128, 64),
        ])
        
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        ])
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.middle(x)
        
        skip_connections = skip_connections[::-1]
        
        for up, skip in zip(self.upconv, skip_connections):
            x = up(x)
            x = torch.cat((x, skip), dim=1)
            x = self.decoder[len(skip_connections) - len(skip_connections[::-1])](x)
        
        return self.final_conv(x)

# Custom Dataset
class AirwayDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = np.load(img_path)
        mask = np.load(mask_path)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            logging.info(f'Train Batch {batch_idx}/{len(train_loader)}: Loss: {loss.item():.6f}')
    
    return total_loss / len(train_loader)

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

# Dice coefficient for evaluation
def dice_coefficient(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Main training loop
def main():
    # Hyperparameters
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 100
    
    # Data directories
    image_dir = 'yolo_inference_output/cropped_iso_images'
    mask_dir = 'yolo_inference_output/cropped_iso_masks'
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset
    dataset = AirwayDataset(image_dir, mask_dir, transform=transform)
    
    # Split dataset
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    # Initialize model, optimizer, and loss function
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Initialize variables to track the best model
    best_val_loss = float('inf')
    best_model_path = 'unet_model_best.pth'

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'New best model saved to {best_model_path}')
        
        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'unet_model_epoch_{epoch+1}.pth')
    
    # Save the final model
    final_model_path = 'unet_model_final.pth'
    torch.save(model.state_dict(), final_model_path)
    logging.info(f'Final model saved to {final_model_path}')

    logging.info("Training completed.")

    # Evaluate the model on the validation set
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = torch.sigmoid(output) > 0.5
            dice_scores.append(dice_coefficient(pred, target).item())
    
    avg_dice = np.mean(dice_scores)
    logging.info(f"Average Dice coefficient on validation set: {avg_dice:.4f}")

    # Visualize some results
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            if i >= 5:  # Visualize 5 examples
                break
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = torch.sigmoid(output) > 0.5
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(data[0, 0].cpu(), cmap='gray')
            axes[0].set_title('Input Image')
            axes[1].imshow(target[0, 0].cpu(), cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[2].imshow(pred[0, 0].cpu(), cmap='gray')
            axes[2].set_title('Prediction')
            plt.show()

if __name__ == '__main__':
    main()