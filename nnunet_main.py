import argparse
from nnunet_preprocessing import preprocess_dataset
from nnunet_inference import inference_on_dataset
from nnunet_train import train_model

def main():
    parser = argparse.ArgumentParser(description="nnU-Net for 2D/3D MRI Segmentation")
    parser.add_argument("--mode", type=str, required=True, choices=['preprocess', 'train', 'inference'], help="Mode of operation")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_path", type=str, help="Path to a pretrained model (for inference)")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of output classes")
    parser.add_argument("--dimensions", type=int, default=3, choices=[2, 3], help="Dimensions of the input data (2D or 3D)")
    parser.add_argument("--input_type", type=str, choices=['nifti', 'png'], required=True, help="Input file type (nifti or png)")
    parser.add_argument("--target_size", type=int, nargs='+', default=[128, 128, 128], help="Target size for preprocessing (provide 2 values for 2D or 3 values for 3D)")
    args = parser.parse_args()

    # Ensure target_size matches dimensions
    if args.dimensions == 2 and len(args.target_size) != 2:
        parser.error("For 2D inputs, target_size should have 2 values")
    elif args.dimensions == 3 and len(args.target_size) != 3:
        parser.error("For 3D inputs, target_size should have 3 values")

    if args.mode == 'preprocess':
        preprocess_dataset(args.input_dir, args.output_dir, target_size=tuple(args.target_size), input_type=args.input_type)
        
    elif args.mode == 'train':
        train_model(args.input_dir, args.output_dir, args.in_channels, args.num_classes, args.dimensions)
        
    elif args.mode == 'inference':
        inference_on_dataset(args.model_path, args.input_dir, args.output_dir, args.in_channels, args.num_classes, args.dimensions)

if __name__ == "__main__":
    main()