# unet_model.py

# use python unet_model.py to create and check the model

import torch
import torch.nn as nn
import torchviz
import logging

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder (downsampling)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # Decoder (upsampling)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)  # 1024 because of skip connection
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)
        
        # Final convolution
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)  # Skip connection
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)  # Skip connection
        x = self.up_conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)  # Skip connection
        x = self.up_conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)  # Skip connection
        x = self.up_conv4(x)
        
        logits = self.outc(x)
        return logits

def save_model_architecture(model, input_size=(1, 1, 256, 256), filename='unet_architecture.png'):
    """
    Save the model architecture as a PNG file.
    """
    x = torch.randn(input_size)
    y = model(x)
    dot = torchviz.make_dot(y, params=dict(model.named_parameters()))
    dot.render(filename, format='png', cleanup=True)
    logging.info(f"Model architecture saved as {filename}")

def check_input_dimension(model, input_size=(1, 1, 256, 256), device=device):
    """
    Check if the input dimension matches the model's expected input.
    """
    try:
        x = torch.randn(input_size).to(device)
        _ = model(x)
        logging.info(f"Input dimension check passed. Model accepts input size: {input_size}")
    except RuntimeError as e:
        logging.error(f"Input dimension mismatch. Error: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = UNet(n_channels=1, n_classes=1)
    
    # Save model architecture
    save_model_architecture(model)
    
    # Check input dimension
    check_input_dimension(model, device=device)