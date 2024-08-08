import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from nnunet_model import create_nnunet
from nnunet_dataset import NNUnetDataset
import os
from tqdm import tqdm

def train_model(train_dir, output_dir, in_channels, num_classes, dimensions, 
                batch_size=4, num_epochs=300, learning_rate=0.001, val_split=0.2):
    os.makedirs(output_dir, exist_ok=True)
    # Create model
    model = create_nnunet(in_channels, num_classes, dimensions)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create dataset
    target_size = (256, 256) if dimensions == 2 else (128, 128, 128)
    full_dataset = NNUnetDataset(train_dir, target_size=target_size, dimensions=dimensions)
    
    # Split dataset into train and validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for batch in train_loader:
                inputs, targets = batch['image'].to(device), batch['mask'].to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                if isinstance(outputs, tuple):
                    main_output = outputs[0]
                else:
                    main_output = outputs
                
                loss = criterion(main_output, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                pbar.update(1)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch['image'].to(device), batch['mask'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")

if __name__ == "__main__":
    train_dir = "path/to/preprocessed/data"
    output_dir = "path/to/output/directory"
    in_channels = 1
    num_classes = 1
    dimensions = 3  # Change this to 2 for 2D inputs
    train_model(train_dir, output_dir, in_channels, num_classes, dimensions)