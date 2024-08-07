import torch
import nibabel as nib
import numpy as np
from nn_unet_model import create_nnunet
from preprocessing import preprocess_case
from postprocessing import postprocess_prediction
import os

def load_model(model_path, in_channels, num_classes):
    model = create_nnunet(in_channels, num_classes, dimensions=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def inference_on_case(model, image_path, mask_path=None, output_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Preprocess the image
    preprocessed_image, _, affine = preprocess_case(image_path, mask_path)
    
    # Prepare input for the model
    input_tensor = torch.from_numpy(preprocessed_image).float().unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Post-process the prediction
    prediction = output.squeeze().cpu().numpy()
    postprocessed_prediction = postprocess_prediction(prediction)
    
    # Save the result
    if output_path:
        nib.save(nib.Nifti1Image(postprocessed_prediction, affine), output_path)
    
    return postprocessed_prediction

def inference_on_dataset(model_path, image_dir, mask_dir, output_dir):
    model = load_model(model_path, in_channels=1, num_classes=1)
    os.makedirs(output_dir, exist_ok=True)
    
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.nii') or image_file.endswith('.nii.gz'):
            image_path = os.path.join(image_dir, image_file)
            mask_path = os.path.join(mask_dir, image_file.replace('image', 'mask'))
            output_path = os.path.join(output_dir, f"prediction_{image_file}")
            
            inference_on_case(model, image_path, mask_path, output_path)
    
    print("Inference completed.")

if __name__ == "__main__":
    model_path = "path/to/trained/model.pth"
    image_dir = "path/to/image/directory"
    mask_dir = "path/to/mask/directory"
    output_dir = "path/to/output/directory"
    inference_on_dataset(model_path, image_dir, mask_dir, output_dir)