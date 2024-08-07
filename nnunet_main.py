import argparse
import os
import torch
from nnunet_model import create_nnunet, save_model_architecture
from preprocessing import preprocess_dataset
from nnunet_inference import inference_on_dataset
from nnunet_postprocessing import postprocess_prediction
import nibabel as nib
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="nnU-Net for 3D MRI Segmentation")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input files (NIfTI or PNG)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save preprocessed data, results, and model diagram")
    parser.add_argument("--model_path", type=str, help="Path to a pretrained model (if not provided, a new model will be created)")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of output classes")
    parser.add_argument("--dimensions", type=int, default=3, choices=[2, 3], help="Dimensions of the input data (2D or 3D)")
    parser.add_argument("--input_size", type=int, nargs=3, default=[128, 128, 128], help="Input size for the model (e.g., 128 128 128)")
    parser.add_argument("--input_type", type=str, choices=['nifti', 'png'], required=True, help="Input file type (nifti or png)")
    args = parser.parse_args()

    # Create output directories
    preprocessed_dir = os.path.join(args.output_dir, "preprocessed")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Preprocess the dataset
    print("Preprocessing the dataset...")
    preprocess_dataset(args.input_dir, preprocessed_dir, args.input_type)

    # Create or load the model
    model = create_nnunet(args.in_channels, args.num_classes, args.dimensions)
    if args.model_path:
        print(f"Loading pretrained model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))
    else:
        print("Creating a new model")

    # Save model architecture diagram
    save_model_architecture(model, tuple([args.in_channels] + args.input_size), args.output_dir)

    # Perform inference
    print("Running inference...")
    inference_on_dataset(model, preprocessed_dir, results_dir, args.input_type)

    # Postprocess the results
    print("Postprocessing the results...")
    for file in os.listdir(results_dir):
        if file.endswith(".nii.gz") or file.endswith(".png"):
            file_path = os.path.join(results_dir, file)
            if file.endswith(".nii.gz"):
                img = nib.load(file_path)
                data = img.get_fdata()
                postprocessed = postprocess_prediction(data)
                nib.save(nib.Nifti1Image(postprocessed, img.affine), file_path)
            else:
                img = Image.open(file_path)
                data = np.array(img)
                postprocessed = postprocess_prediction(data)
                Image.fromarray(postprocessed.astype(np.uint8)).save(file_path)

    print(f"Processing complete. Results saved in {results_dir}")

if __name__ == "__main__":
    main()