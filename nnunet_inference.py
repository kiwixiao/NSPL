import torch
import numpy as np
from nnunet_model import create_nnunet
from nnunet_postprocessing import postprocess_prediction
import nibabel as nib
from PIL import Image
import os
from scipy.ndimage import zoom

def load_model(model_path, in_channels, num_classes, dimensions):
    model = create_nnunet(in_channels, num_classes, dimensions)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def pad_image(image, target_size):
    w, h = image.size
    target_w, target_h = target_size
    pad_w = max(target_w - w, 0)
    pad_h = max(target_h - h, 0)
    padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
    padded_image = Image.new(image.mode, (max(w, target_w), max(h, target_h)), color=0)
    padded_image.paste(image, (padding[0], padding[1]))
    return padded_image

def resize_to_original(prediction, original_shape):
    if not isinstance(prediction, np.ndarray):
        prediction = np.array(prediction)
    zoom_factors = np.array(original_shape) / np.array(prediction.shape)
    resized = zoom(prediction, zoom_factors, order=0)
    return resized

def inference_on_case(model, image_path, output_path, dimensions, target_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if dimensions == 3:
        image_nii = nib.load(image_path)
        image = image_nii.get_fdata()
        original_shape = image.shape
        image = zoom(image, [t / s for t, s in zip(target_size, image.shape)], order=3)
        image = (image - image.mean()) / image.std()
        image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    else:
        image = Image.open(image_path).convert('L')
        original_shape = image.size
        image = pad_image(image, target_size)
        image = np.array(image)
        image = (image - image.mean()) / (image.std() + 1e-8)
        image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
    
    prediction = torch.sigmoid(output).squeeze().cpu().numpy()
    prediction = postprocess_prediction(prediction)
    
    if dimensions == 3:
        prediction = resize_to_original(prediction, original_shape)
        nib.save(nib.Nifti1Image(prediction, image_nii.affine), output_path)
    else:
        prediction = (prediction * 255).astype(np.uint8)
        prediction_image = Image.fromarray(prediction)
        prediction_image = prediction_image.resize(original_shape, Image.NEAREST)
        prediction_image.save(output_path)
    
    return prediction

def inference_on_dataset(model_path, input_dir, output_dir, in_channels=1, num_classes=1, dimensions=3):
    target_size = (64, 64) if dimensions == 2 else (128, 128, 128)
    model = load_model(model_path, in_channels, num_classes, dimensions)
    os.makedirs(output_dir, exist_ok=True)
    
    file_extension = '.nii.gz' if dimensions == 3 else '.png'
    
    for image_file in os.listdir(input_dir):
        if image_file.endswith(file_extension):
            image_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, f"prediction_{image_file}")
            
            inference_on_case(model, image_path, output_path, dimensions, target_size)
    
    print("Inference completed.")

if __name__ == "__main__":
    model_path = "path/to/trained/model.pth"
    input_dir = "path/to/input/directory"
    output_dir = "path/to/output/directory"
    in_channels = 1
    num_classes = 1
    dimensions = 3  # Change this to 2 for 2D inputs
    inference_on_dataset(model_path, input_dir, output_dir, in_channels, num_classes, dimensions)