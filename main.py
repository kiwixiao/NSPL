import argparse
import torch
from models import UNetResNet, nnUNet, SimpleUNet
from train import train_kfold
from inference import infer
from utils.io_utils import load_dataset
import logging
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="MRI Segmentation Pipeline")
    parser.add_argument('--mode', choices=['train', 'infer'], required=True)
    parser.add_argument('--models', nargs='+', choices=['simple_unet', 'nnunet', 'unet_resnet34', 'unet_resnet50', 'unet_resnet101', 'unet_resnext50'], default=['simple_unet'], help="Models to train")
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', default='unet_model_output', help="Base output directory")
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256], help="Image size (height, width)")
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    dataset = load_dataset(args.data_path, img_size=tuple(args.img_size))

    # Create the base output directory
    os.makedirs(args.output_path, exist_ok=True)

    if args.mode == 'train':
        for model_name in args.models:
            logging.info(f"Training model: {model_name}")
            if model_name == 'simple_unet':
                model_class = lambda n_channels, n_classes: SimpleUNet(n_channels, n_classes)
            elif model_name == 'nnunet':
                model_class = lambda n_channels, n_classes: nnUNet(n_channels, n_classes)
            elif model_name.startswith('unet_resnet') or model_name.startswith('unet_resnext'):
                backbone = model_name.split('_', 1)[1]
                model_class = lambda n_channels, n_classes: UNetResNet(n_channels, n_classes, backbone=backbone)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_output_dir = os.path.join(args.output_path, f"{model_name}_{timestamp}")
            os.makedirs(model_output_dir, exist_ok=True)
            
            train_kfold(model_class, dataset, args.num_epochs, device, n_splits=args.n_splits, 
                        img_size=tuple(args.img_size), output_dir=model_output_dir, model_name=model_name)
    
    elif args.mode == 'infer':
        model_name = args.models[0]
        if model_name == 'simple_unet':
            model = SimpleUNet(n_channels=1, n_classes=1).to(device)
        elif model_name == 'nnunet':
            model = nnUNet(n_channels=1, n_classes=1).to(device)
        elif model_name.startswith('unet_resnet') or model_name.startswith('unet_resnext'):
            backbone = model_name.split('_', 1)[1]
            model = UNetResNet(n_channels=1, n_classes=1, backbone=backbone).to(device)
        
        # Assume the latest model is the one to use
        model_dirs = [d for d in os.listdir(args.output_path) if d.startswith(model_name)]
        latest_model_dir = max(model_dirs, key=lambda x: x.split('_')[-1])
        model_path = os.path.join(args.output_path, latest_model_dir, 'best_model.pth')
        model.load_state_dict(torch.load(model_path))
        
        inference_output_dir = os.path.join(args.output_path, f"inference_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(inference_output_dir, exist_ok=True)
        infer(model, args.data_path, inference_output_dir, device, img_size=tuple(args.img_size))
    
    else:
        raise ValueError("Invalid mode choice")

if __name__ == '__main__':
    main()