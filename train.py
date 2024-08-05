import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from models import SimpleUNet, UNetResNet
from utils.io_utils import load_dataset
from preprocessing.data_preparation import create_kfolds
from utils.visualization import plot_training_curves
import logging
import os
from tqdm import tqdm, trange

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + 1) / (inputs.sum() + targets.sum() + 1)
        
        return bce_loss + dice_loss

def train_fold(model, train_loader, val_loader, num_epochs, device, output_dir, model_name, patience=10):
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    epoch_bar = trange(num_epochs, desc="Epochs")
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} training", leave=False)
        for batch_idx, (images, masks) in enumerate(batch_bar):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            if outputs.size() != masks.size():
                logging.error(f"Size mismatch: outputs {outputs.size()}, masks {masks.size()}")
                raise ValueError(f"Output size {outputs.size()} doesn't match target size {masks.size()}")
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            batch_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        epoch_bar.set_postfix({"Train Loss": f"{train_loss:.4f}", "Val Loss": f"{val_loss:.4f}"})
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_path = os.path.join(output_dir, f'best_model_{model_name}.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                logging.info("Early stopping")
                break

    return train_losses, val_losses

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    
    val_bar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, masks in val_bar:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            val_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    return val_loss / len(val_loader)

def train_kfold(model_class, dataset, num_epochs, device, n_splits=5, img_size=(256, 256), output_dir='./', model_name='model'):
    kfolds = create_kfolds(dataset, n_splits=n_splits)
    
    fold_bar = tqdm(enumerate(kfolds), total=n_splits, desc="Folds")
    for fold, (train_idx, val_idx) in fold_bar:
        logging.info(f"Training fold {fold+1}/{n_splits}")
        fold_bar.set_postfix({"Fold": f"{fold+1}/{n_splits}"})
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=16)
        
        model = model_class(n_channels=1, n_classes=1).to(device)
        
        fold_output_dir = os.path.join(output_dir, f'fold_{fold+1}')
        os.makedirs(fold_output_dir, exist_ok=True)
        
        train_losses, val_losses = train_fold(model, train_loader, val_loader, num_epochs, device, fold_output_dir, model_name)
        
        plot_training_curves(train_losses, val_losses, os.path.join(fold_output_dir, f'loss_plot_{model_name}.png'))
    
    logging.info("K-fold cross-validation completed")