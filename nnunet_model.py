import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from graphviz import Digraph
import os

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dimensions):
        super(ConvBlock, self).__init__()
        if dimensions == 2:
            self.conv = nn.Conv2d
            self.norm = nn.InstanceNorm2d
        elif dimensions == 3:
            self.conv = nn.Conv3d
            self.norm = nn.InstanceNorm3d
        
        self.block = nn.Sequential(
            self.conv(in_channels, out_channels, 3, padding=1),
            self.norm(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            self.conv(out_channels, out_channels, 3, padding=1),
            self.norm(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class FlexibleUNet(nn.Module):
    def __init__(self, in_channels, num_classes, dimensions=3):
        super(FlexibleUNet, self).__init__()
        self.dimensions = dimensions
        self.num_classes = num_classes
        
        if dimensions == 2:
            self.conv = nn.Conv2d
            self.maxpool = nn.MaxPool2d(2)
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif dimensions == 3:
            self.conv = nn.Conv3d
            self.maxpool = nn.MaxPool3d(2)
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        self.down_conv1 = ConvBlock(in_channels, 32, dimensions)
        self.down_conv2 = ConvBlock(32, 64, dimensions)
        self.down_conv3 = ConvBlock(64, 128, dimensions)
        self.down_conv4 = ConvBlock(128, 256, dimensions)
        self.down_conv5 = ConvBlock(256, 320, dimensions)
        
        self.up_conv4 = ConvBlock(320 + 256, 256, dimensions)
        self.up_conv3 = ConvBlock(256 + 128, 128, dimensions)
        self.up_conv2 = ConvBlock(128 + 64, 64, dimensions)
        self.up_conv1 = ConvBlock(64 + 32, 32, dimensions)
        
        self.final_conv = self.conv(32, num_classes, kernel_size=1)
        
        # Deep supervision outputs
        self.deep_sup3 = self.conv(128, num_classes, kernel_size=1)
        self.deep_sup2 = self.conv(64, num_classes, kernel_size=1)
        self.deep_sup1 = self.conv(32, num_classes, kernel_size=1)
    
    def forward(self, x):
        conv1 = self.down_conv1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.down_conv2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.down_conv3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.down_conv4(x)
        x = self.maxpool(conv4)
        
        x = self.down_conv5(x)
        
        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.up_conv4(x)
        
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.up_conv3(x)
        deep_sup3 = self.deep_sup3(x)
        
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv2(x)
        deep_sup2 = self.deep_sup2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv1(x)
        deep_sup1 = self.deep_sup1(x)
        
        out = self.final_conv(x)
        
        if self.num_classes == 1:
            out = torch.sigmoid(out)
            if self.training:
                deep_sup1 = torch.sigmoid(deep_sup1)
                deep_sup2 = torch.sigmoid(deep_sup2)
                deep_sup3 = torch.sigmoid(deep_sup3)
        else:
            out = F.softmax(out, dim=1)
            if self.training:
                deep_sup1 = F.softmax(deep_sup1, dim=1)
                deep_sup2 = F.softmax(deep_sup2, dim=1)
                deep_sup3 = F.softmax(deep_sup3, dim=1)
        
        if self.training:
            return out, deep_sup1, deep_sup2, deep_sup3
        else:
            return out

def create_nnunet(in_channels, num_classes, dimensions=3):
    return FlexibleUNet(in_channels, num_classes, dimensions)

def save_model_architecture(model, input_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a dummy input tensor
    x = torch.randn(1, *input_size)
    
    # Generate the model graph
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    
    # Customize the graph
    dot.attr(rankdir='TB', size='12,12')
    dot.attr('node', shape='record')
    
    for node in dot.graph.get_nodes():
        if node.get_name().startswith('node'):
            label = node.get_attributes()['label']
            if 'ConvBlock' in label or 'Conv' in label:
                parts = label.split('\\n')
                if len(parts) > 1:
                    shape_info = parts[1].strip('()')
                    input_shape, output_shape = shape_info.split(' -> ')
                    new_label = f"{parts[0]}|{{{input_shape}|{output_shape}}}"
                    node.set_attributes({'label': new_label})
    
    # Save the diagram as a PNG file
    dot.render(os.path.join(output_dir, 'model_architecture'), format='png', cleanup=True)
    print(f"Model architecture diagram saved to {output_dir}/model_architecture.png")

if __name__ == "__main__":
    # Example usage
    in_channels = 1
    num_classes = 1
    dimensions = 3
    input_size = (in_channels, 128, 128, 128)  # Example input size (adjust as needed)
    
    model = create_nnunet(in_channels, num_classes, dimensions)
    save_model_architecture(model, input_size, output_dir='model_diagrams')