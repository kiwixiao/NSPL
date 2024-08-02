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
from scipy.ndimage import zoom

def select_file(title):
    root = tk.Tk()
    root.withdraw()
    initial_dir = os.getcwd()
    file_path = filedialog.askopenfilename(title=title, initialdir=initial_dir)
    return file_path

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = YOLO(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model

def resample_volume(volume, target_spacing=(0.6, 0.6, 0.6)):
    current_spacing = volume.header.get_zooms()[:3]
    current_shape = volume.shape[:3]
    scale_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
    new_shape = np.round(current_shape * np.array(scale_factors)).astype(int)
    resampled = zoom(volume.get_fdata(), scale_factors, order=3, mode='constant')
    return resampled

def normalize_image(image):
    centered = (image - np.mean(image)) / np.std(image)
    min_val, max_val = np.min(centered), np.max(centered)
    normalized = (centered - min_val) / (max_val - min_val)
    return normalized

def preprocess_slice(slice_img, input_size=(640, 640)):
    slice_img_normalized = normalize_image(slice_img)
    slice_img_resized = cv2.resize(slice_img_normalized, input_size)
    slice_img_rgb = np.stack([slice_img_resized] * 3, axis=-1)  # Convert to 3-channel
    return slice_img_rgb

def inference_on_mri(model, mri_path):
    print(f"Performing inference on {mri_path}")
    mri_img = nib.load(mri_path)
    
    # Reorient to canonical orientation
    mri_canonical = nib.as_closest_canonical(mri_img)
    
    # Resample to target spacing
    mri_resampled = resample_volume(mri_canonical)
    
    results = []
    for i in range(mri_resampled.shape[1]):  # Assuming coronal slices
        slice_img = mri_resampled[:,i,:]
        processed_slice = preprocess_slice(slice_img)
        result = model(processed_slice)
        results.append(result)
    
    print(f"Inference completed on {len(results)} slices")
    return results, mri_resampled

def visualize_results(mri, results, output_path, title):
    total_slices = mri.shape[1]  # Assuming coronal slices
    step = max(1, total_slices // 25)  # We want 25 slices for a 5x5 grid
    
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    fig.suptitle(title, fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        slice_idx = i * step
        if slice_idx >= total_slices:
            ax.axis('off')
            continue
        
        slice_img = mri[:,slice_idx,:]
        ax.imshow(slice_img, cmap='gray')
        
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
    trained_model_path = select_file("Select your trained model checkpoint file")
    if not trained_model_path:
        print("No trained model file selected. Exiting.")
        return

    mri_path = select_file("Select the MRI image file")
    if not mri_path:
        print("No MRI file selected. Exiting.")
        return

    trained_model = load_model(trained_model_path)
    pretrained_model = YOLO('yolov8m.pt')
    
    trained_results, mri_resampled = inference_on_mri(trained_model, mri_path)
    pretrained_results, _ = inference_on_mri(pretrained_model, mri_path)
    
    inference_dir = os.path.join(os.getcwd(), 'inference_results')
    os.makedirs(inference_dir, exist_ok=True)
    
    cwd = os.getcwd()
    rel_path = os.path.relpath(os.path.dirname(trained_model_path), cwd)
    model_folder = rel_path.replace('/', '_').replace('\\', '_')
    if model_folder == '.':
        model_folder = 'current_directory'
    elif model_folder.startswith('_'):
        model_folder = model_folder[1:]

    trained_output_path = os.path.join(inference_dir, f'{model_folder}_trained_model_results.png')
    visualize_results(mri_resampled, trained_results, trained_output_path, f"Trained Model ({model_folder}) Inference Results")
    
    pretrained_output_path = os.path.join(inference_dir, f'Pretrained_model_results.png')
    visualize_results(mri_resampled, pretrained_results, pretrained_output_path, f"Pre-trained YOLOv8m vs {model_folder} Inference Results")

if __name__ == "__main__":
    main()