# In nnunet_main.py

import argparse
from nnunet_preprocessing import preprocess_dataset
from nnunet_inference import inference_on_dataset
from nnunet_train import train_model

def main():
    parser = argparse.ArgumentParser(description="nnU-Net for 3D MRI Segmentation")
    parser.add_argument("--mode", type=str, required=True, choices=['preprocess', 'train', 'inference'], help="Mode of operation")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_path", type=str, help="Path to a pretrained model (for inference)")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of output classes")
    parser.add_argument("--dimensions", type=int, default=3, choices=[2, 3], help="Dimensions of the input data (2D or 3D)")
    parser.add_argument("--input_type", type=str, choices=['nifti', 'png'], required=True, help="Input file type (nifti or png)")
    args = parser.parse_args()

    if args.mode == 'preprocess':
        preprocess_dataset(args.input_dir, args.output_dir, args.input_type)
    elif args.mode == 'train':
        train_model(args.input_dir, args.output_dir, args.in_channels, args.num_classes, args.dimensions)
    elif args.mode == 'inference':
        inference_on_dataset(args.model_path, args.input_dir, args.output_dir, args.input_type)

if __name__ == "__main__":
    main()