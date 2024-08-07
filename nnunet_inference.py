import torch
import nibabel as nib
import numpy as np
from nnunet_model import create_nnunet
from nnunet_preprocessing import preprocess_case
from nnunet_postprocessing import postprocess_prediction
import os
from PIL import Image
from scipy.ndimage import zoom

def load_model(model_path, in_channels, num_classes):
    model = create_nnunet(in_channels, num_classes, dimensions=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def inference_on_case(model, image_path, mask_path=None, output_path=None, input_type='nifti'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Preprocess the image
    preprocessed_image, _, affine = preprocess_case(image_path, mask_path, input_type=input_type)
    
    # Prepare input for the model
    input_tensor = torch.from_numpy(preprocessed_image).float().unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Post-process the prediction
    if isinstance(output, tuple):
        prediction = output[0]  # Take the main output, not the deep supervision outputs
    else:
        prediction = output
    
    prediction = prediction.squeeze().cpu().numpy()
    postprocessed_prediction = postprocess_prediction(prediction)
    
    # Save the result
    if output_path:
        if input_type == 'nifti':
            # Resize the prediction back to the original image size
            original_image = nib.load(image_path)
            original_shape = original_image.shape
            resized_prediction = resize_to_original(postprocessed_prediction, original_shape)
            
            # Save as NIfTI with the original affine
            nib.save(nib.Nifti1Image(resized_prediction, affine), output_path)
        else:
            Image.fromarray(postprocessed_prediction.astype(np.uint8)).save(output_path)
    
    return postprocessed_prediction

def resize_to_original(prediction, original_shape):
    # Ensure prediction is a numpy array
    if not isinstance(prediction, np.ndarray):
        prediction = np.array(prediction)
    
    # Calculate zoom factors
    zoom_factors = np.array(original_shape) / np.array(prediction.shape)
    
    # Use scipy's zoom function for resizing
    resized = zoom(prediction, zoom_factors, order=0)  # order=0 for nearest neighbor interpolation
    
    return resized

def inference_on_dataset(model, input_dir, output_dir, input_type='nifti'):
    os.makedirs(output_dir, exist_ok=True)
    
    for image_file in os.listdir(input_dir):
        if (input_type == 'nifti' and (image_file.endswith('.nii') or image_file.endswith('.nii.gz'))) or \
           (input_type == 'png' and image_file.endswith('.png') and 'mask' not in image_file):
            image_path = os.path.join(input_dir, image_file)
            mask_file = image_file.replace('image', 'mask') if input_type == 'png' else image_file.replace('image', 'mask')
            mask_path = os.path.join(input_dir, mask_file)
            output_path = os.path.join(output_dir, f"prediction_{image_file}")
            
            inference_on_case(model, image_path, mask_path, output_path, input_type)
    
    print("Inference completed.")

if __name__ == "__main__":
    model_path = "path/to/trained/model.pth"
    input_dir = "path/to/input/directory"
    output_dir = "path/to/output/directory"
    input_type = "nifti"  # or "png"
    model = load_model(model_path, in_channels=1, num_classes=1)
    inference_on_dataset(model, input_dir, output_dir, input_type)