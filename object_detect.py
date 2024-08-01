import os
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import glob
import torch # maks sure impot torch for checking GPU availability
import matplotlib.pyplot as plt

def find_objects(mask):
    _, binary_mask = cv2.threshold(mask.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x = (x + w / 2) / mask.shape[1]
        center_y = (y + h / 2) / mask.shape[0]
        width = w / mask.shape[1]
        height = h / mask.shape[0]
        objects.append([0, center_x, center_y, width, height])  # 0 is the class index for airway
    
    return objects

def save_image(img, filename):
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    Image.fromarray(img).save(filename)
    print(f"Saved image: {filename}")

def save_yolo_annotation(objects, filename):
    with open(filename, 'w') as f:
        for obj in objects:
            f.write(' '.join(map(str, obj)) + '\n')
    print(f"Saved annotation: {filename}")

def process_mri_to_yolo(mri_path, mask_path, output_dir, subject_id):
    print(f"Processing MRI: {mri_path}")
    print(f"Processing Mask: {mask_path}")
    mri = nib.load(mri_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    
    print(f"MRI shape: {mri.shape}")
    print(f"Mask shape: {mask.shape}")
    
    for i in range(mri.shape[2]):
        slice_img = mri[:,:,i]
        slice_mask = mask[:,:,i]
        
        objects = find_objects(slice_mask)
        
        if objects:
            img_filename = f"{output_dir}/images/{subject_id}_{i:03d}.png"
            save_image(slice_img, img_filename)
            
            txt_filename = f"{output_dir}/labels/{subject_id}_{i:03d}.txt"
            save_yolo_annotation(objects, txt_filename)
        else:
            print(f"No objects found in slice {i} of {subject_id}")

def process_all_mri_data(image_dir, mask_dir, output_dir):
    print(f"Processing all MRI data:")
    print(f"Image directory: {image_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Output directory: {output_dir}")

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    print(f"Contents of image directory:")
    print(os.listdir(image_dir))
    print(f"Contents of mask directory:")
    print(os.listdir(mask_dir))

    # Find all MRI files
    mri_files = glob.glob(os.path.join(image_dir, "OSAMRI*_mri.nii")) + glob.glob(os.path.join(image_dir, "OSAMRI*_mri.nii.gz"))

    for mri_file in mri_files:
        print(f"Processing MRI file: {mri_file}")
        
        # Extract subject ID
        subject_id = os.path.basename(mri_file).split('_')[0]
        
        # Find corresponding mask file
        mask_pattern = os.path.join(mask_dir, f"{subject_id}*_mask.nii*")
        mask_files = glob.glob(mask_pattern)
        
        if mask_files:
            mask_file = mask_files[0]  # Take the first matching mask file
            print(f"Found corresponding mask: {mask_file}")
            process_mri_to_yolo(mri_file, mask_file, output_dir, subject_id)
        else:
            print(f"Warning: No matching mask found for {mri_file}")
            print(f"Checked pattern: {mask_pattern}")

    print(f"Finished processing all MRI data")

def split_data(output_dir, train_ratio=0.8):
    print(f"Splitting data in {output_dir}")
    image_files = [f for f in os.listdir(f"{output_dir}/images") if f.endswith('.png')]
    train_files, val_files = train_test_split(image_files, train_size=train_ratio, random_state=42)

    os.makedirs(f"{output_dir}/train/images", exist_ok=True)
    os.makedirs(f"{output_dir}/train/labels", exist_ok=True)
    os.makedirs(f"{output_dir}/val/images", exist_ok=True)
    os.makedirs(f"{output_dir}/val/labels", exist_ok=True)

    for file in train_files:
        os.rename(f"{output_dir}/images/{file}", f"{output_dir}/train/images/{file}")
        os.rename(f"{output_dir}/labels/{file.replace('.png', '.txt')}", f"{output_dir}/train/labels/{file.replace('.png', '.txt')}")
        print(f"Moved {file} to train set")

    for file in val_files:
        os.rename(f"{output_dir}/images/{file}", f"{output_dir}/val/images/{file}")
        os.rename(f"{output_dir}/labels/{file.replace('.png', '.txt')}", f"{output_dir}/val/labels/{file.replace('.png', '.txt')}")
        print(f"Moved {file} to validation set")

    os.rmdir(f"{output_dir}/images")
    os.rmdir(f"{output_dir}/labels")
    print("Data split completed")

def create_data_yaml(output_dir):
    print(f"Creating data.yaml in {output_dir}")
    current_dir = os.getcwd()
    data = {
        'train': os.path.join(current_dir, output_dir, 'train', 'images'),
        'val': os.path.join(current_dir, output_dir, 'val', 'images'),
        'nc': 1,
        'names': ['airway']
    }

    with open(f"{output_dir}/data.yaml", 'w') as f:
        yaml.dump(data, f)
    print("data.yaml created")

def train_yolo_model(output_dir, epochs=1):
    print(f"Training YOLOv8 model with data from {output_dir}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load a model
    model = YOLO('yolov8n.yaml')  # build a new model from YAML
    
    data_yaml_path = os.path.join(os.getcwd(), output_dir, 'data.yaml')
    
    try:
        # Train the model
        results = model.train(data=data_yaml_path, epochs=epochs, imgsz=640, device=device)
        
        print("Training completed successfully")
        
        # The model is automatically saved after training
        print(f"Model saved at: {results.save_dir}")
        
        # You can also manually save the model if needed
        model.save('saved_model/best.pt')
        print("Model manually saved at: saved_model/best.pt")
        
        # Print some training metrics
        print(f"Best mAP50: {results.best_map50}")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
    
    return model

def inference_on_mri(model, mri_path):
    print(f"Performing inference on {mri_path}")
    mri = nib.load(mri_path).get_fdata()
    
    results = []
    for i in range(mri.shape[2]):
        slice_img = mri[:,:,i]
        slice_img = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
        result = model(slice_img)
        results.append(result)
    
    print("Inference completed")
    return results

def visualize_results(mri, results):
    print("Visualizing results")
    for i, result in enumerate(results):
        plt.figure(figsize=(10, 10))
        plt.imshow(mri[:,:,i], cmap='gray')
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=2))
        
        plt.title(f"Slice {i}")
        plt.axis('off')
        plt.show()
    print("Visualization completed")

if __name__ == "__main__":
    image_dir = "./images"
    mask_dir = "./masks"
    output_dir = "yolo_data"

    print(f"Starting script execution")
    print(f"Working directory: {os.getcwd()}")
    print(f"Image directory: {image_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Output directory: {output_dir}")

    try:
        # Check if directories exist
        for dir_path in [image_dir, mask_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")

        # Process MRI data
        process_all_mri_data(image_dir, mask_dir, output_dir)

        # Check if output directories were created
        for subdir in ['train/images', 'train/labels', 'val/images', 'val/labels']:
            full_path = os.path.join(output_dir, subdir)
            if not os.path.exists(full_path):
                print(f"path {full_path} not exist")
                print("will create them")
                os.makedirs(full_path, exist_ok=True)
                #raise FileNotFoundError(f"Expected directory not found: {full_path}")
            else:
                print(f"Full path for yolo training: {full_path}")

        # Split data
        split_data(output_dir)

        # Create data.yaml
        create_data_yaml(output_dir)

        # Train model
        model = train_yolo_model(output_dir)

        print("Script execution completed successfully")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()