import os
import nibabel as nib
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import matplotlib.patches as patches

def select_file(title):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    initial_dir = os.getcwd()  # Set current working directory as default
    file_path = filedialog.askopenfilename(title=title, initialdir=initial_dir)
    return file_path

def load_model(model_path):
    """Load the trained YOLO model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = YOLO(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model

def preprocess_slice(slice_img, input_size=(640, 640)):
    """Preprocess a single MRI slice for inference."""
    slice_img = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
    slice_img_rgb = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2RGB)
    slice_img_resized = cv2.resize(slice_img_rgb, input_size)
    return slice_img_resized

def inference_on_mri(model, mri_path):
    """Perform inference on all slices of an MRI image."""
    print(f"Performing inference on {mri_path}")
    mri = nib.load(mri_path).get_fdata()
    
    results = []
    for i in range(mri.shape[2]):
        slice_img = mri[:,:,i]
        processed_slice = preprocess_slice(slice_img)
        result = model(processed_slice)
        results.append(result)
    
    print(f"Inference completed on {len(results)} slices")
    return results, mri

def visualize_results(mri, results, output_path, title):
    """Visualize and plot the inference results as a 5x5 grid with bounding boxes."""
    total_slices = mri.shape[2]
    step = max(1, total_slices // 25)  # We want 25 slices for a 5x5 grid
    
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    fig.suptitle(title, fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        slice_idx = i * step
        if slice_idx >= total_slices:
            ax.axis('off')
            continue
        
        ax.imshow(mri[:,:,slice_idx], cmap='gray')
        
        result = results[slice_idx][0]  # Get the first (and only) image result
        boxes = result.boxes.xyxy.cpu().numpy()
        
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        ax.set_title(f"Slice {slice_idx}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

def main():
    # Ask for trained model path using file dialog
    trained_model_path = select_file("Select your trained model checkpoint file")
    if not trained_model_path:
        print("No trained model file selected. Exiting.")
        return

    # Ask for MRI image path using file dialog
    mri_path = select_file("Select the MRI image file")
    if not mri_path:
        print("No MRI file selected. Exiting.")
        return

    # Load the trained model
    trained_model = load_model(trained_model_path)
    
    # Load the pre-trained YOLOv8m model
    pretrained_model = YOLO('yolov8m.pt')
    
    # Perform inference with trained model
    trained_results, mri = inference_on_mri(trained_model, mri_path)
    
    # Perform inference with pre-trained model
    pretrained_results, _ = inference_on_mri(pretrained_model, mri_path)
    
    # Create inference_results folder
    inference_dir = os.path.join(os.getcwd(), 'inference_results')
    os.makedirs(inference_dir, exist_ok=True)
    
    # Get the model folder path relative to the current working directory
    cwd = os.getcwd()
    rel_path = os.path.relpath(os.path.dirname(trained_model_path), cwd)
    model_folder = rel_path.replace('/', '_').replace('\\', '_')
    if model_folder == '.':
        model_folder = 'current_directory'
    elif model_folder.startswith('_'):
        model_folder = model_folder[1:]  # Remove leading underscore if present

    # Visualize results for trained model
    trained_output_path = os.path.join(inference_dir, f'{model_folder}_trained_model_results.png')
    visualize_results(mri, trained_results, trained_output_path, f"Trained Model ({model_folder}) Inference Results")
    
    # Visualize results for pre-trained model
    pretrained_output_path = os.path.join(inference_dir, f'Pretrained_model_results.png')
    visualize_results(mri, pretrained_results, pretrained_output_path, f"Pre-trained YOLOv8m vs {model_folder} Inference Results")

# This allows the script to be run directly
if __name__ == "__main__":
    main()