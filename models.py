import torch
import torch.nn as nn
import torchvision.models as models
import logging

class UNetResNet(nn.Module):
    def __init__(self, n_channels, n_classes, backbone='resnet34'):
        super(UNetResNet, self).__init__()
        
        if backbone == 'resnet34':
            self.encoder = models.resnet34(pretrained=True)
        elif backbone == 'resnet50':
            self.encoder = models.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            self.encoder = models.resnet101(pretrained=True)
        elif backbone == 'resnext50':
            self.encoder = models.resnext50_32x4d(pretrained=True)
        else:
            raise ValueError("Unsupported backbone")
        
        self.encoder.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.decoder4 = self._decoder_block(512, 256)
        self.decoder3 = self._decoder_block(256, 128)
        self.decoder2 = self._decoder_block(128, 64)
        self.decoder1 = self._decoder_block(64, 32)
        
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.encoder.conv1(x)
        x1 = self.encoder.bn1(x1)
        x1 = self.encoder.relu(x1)
        x1 = self.encoder.maxpool(x1)
        
        x2 = self.encoder.layer1(x1)
        x3 = self.encoder.layer2(x2)
        x4 = self.encoder.layer3(x3)
        x5 = self.encoder.layer4(x4)
        
        x = self.decoder4(x5)
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)
        
        x = self.final_conv(x)
        x = self.upsample(x)  # Upsample to match input size
        
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class nnUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(nnUNet, self).__init__()
        
        self.encoder1 = ConvBlock(n_channels, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)
        self.encoder4 = ConvBlock(128, 256)
        self.encoder5 = ConvBlock(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        self.decoder4 = ConvBlock(512 + 256, 256)
        self.decoder3 = ConvBlock(256 + 128, 128)
        self.decoder2 = ConvBlock(128 + 64, 64)
        self.decoder1 = ConvBlock(64 + 32, 32)
        
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        logging.debug(f"Input shape: {x.shape}")
        
        e1 = self.encoder1(x)
        logging.debug(f"Encoder 1 output shape: {e1.shape}")
        e2 = self.encoder2(self.pool(e1))
        logging.debug(f"Encoder 2 output shape: {e2.shape}")
        e3 = self.encoder3(self.pool(e2))
        logging.debug(f"Encoder 3 output shape: {e3.shape}")
        e4 = self.encoder4(self.pool(e3))
        logging.debug(f"Encoder 4 output shape: {e4.shape}")
        e5 = self.encoder5(self.pool(e4))
        logging.debug(f"Encoder 5 output shape: {e5.shape}")
        
        d4 = self.upconv4(e5)
        logging.debug(f"Decoder 4 upconv output shape: {d4.shape}")
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)
        logging.debug(f"Decoder 4 output shape: {d4.shape}")
        
        d3 = self.upconv3(d4)
        logging.debug(f"Decoder 3 upconv output shape: {d3.shape}")
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        logging.debug(f"Decoder 3 output shape: {d3.shape}")
        
        d2 = self.upconv2(d3)
        logging.debug(f"Decoder 2 upconv output shape: {d2.shape}")
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        logging.debug(f"Decoder 2 output shape: {d2.shape}")
        
        d1 = self.upconv1(d2)
        logging.debug(f"Decoder 1 upconv output shape: {d1.shape}")
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        logging.debug(f"Decoder 1 output shape: {d1.shape}")
        
        out = self.final_conv(d1)
        logging.debug(f"Final output shape: {out.shape}")
        
        return out

# New simple UNet model
class SimpleUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SimpleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

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

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


#%%
def check_model_sizes(model, input_size=(1, 1, 256, 256)):
    logging.info(f"Checking model sizes with input size: {input_size}")
    x = torch.randn(input_size)
    model(x)
    
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total number of parameters: {total_params}")
    
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total number of trainable parameters: {total_trainable_params}")

def check_model_sizes(model, input_size=(1, 1, 256, 256)):
    logging.info(f"Checking model sizes with input size: {input_size}")
    x = torch.randn(input_size)
    model(x)
    
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total number of parameters: {total_params}")
    
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total number of trainable parameters: {total_trainable_params}")

def print_model_structure(model, input_size=(1, 1, 256, 256)):
    def hook(module, input, output):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        input_shape = input[0].shape if isinstance(input, tuple) else input.shape
        output_shape = output.shape if isinstance(output, torch.Tensor) else (output,)
        logging.info(f"{class_name:<20} input: {input_shape}, output: {output_shape}")

    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))

    try:
        model(torch.randn(input_size))
    except Exception as e:
        logging.error(f"Error during model structure print: {str(e)}")
    finally:
        for h in hooks:
            h.remove()