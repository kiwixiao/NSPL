import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

def load_and_check_image(image_path):
    try:
        with Image.open(image_path) as img:
            print(f"Processing: {os.path.basename(image_path)}")
            print(f"Image format: {img.format}")
            print(f"Image mode: {img.mode}")
            print(f"Image size: {img.size}")
            
            img_rgb = img.convert('RGB')
            img_array = np.array(img_rgb)
            
            print(f"Array shape: {img_array.shape}")
            print(f"Array dtype: {img_array.dtype}")
            print(f"Array min and max values: {img_array.min()}, {img_array.max()}")
            
            return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def process_and_save_image(model, image_path, output_folder, combined_folder):
    input_image = load_and_check_image(image_path)
    
    if input_image is not None:
        results = model(input_image)
        
        # Original prediction plot
        plt.figure(figsize=(10, 10))
        plt.imshow(input_image)
        plt.axis('off')
        
        all_boxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                all_boxes.append([x1, y1, x2, y2])
                
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
                
                plt.text(x1, y1, f'Airway: {conf:.2f}', color='red', fontsize=12, 
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0))
        
        plt.title('MRI Airway Detection Results')
        plt.tight_layout()
        
        output_path = os.path.join(output_folder, f"pred_{os.path.basename(image_path)}")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved prediction to: {output_path}")
        
        # Combined bounding box plot
        if all_boxes:
            all_boxes = np.array(all_boxes)
            x1, y1 = all_boxes[:, :2].min(axis=0)
            x2, y2 = all_boxes[:, 2:].max(axis=0)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(input_image)
            plt.axis('off')
            
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='blue', linewidth=2)
            plt.gca().add_patch(rect)
            
            plt.text(x1, y1, 'Combined', color='blue', fontsize=12, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0))
            
            plt.title('Combined MRI Airway Detection Results')
            plt.tight_layout()
            
            combined_output_path = os.path.join(combined_folder, f"combined_pred_{os.path.basename(image_path)}")
            plt.savefig(combined_output_path)
            plt.close()
            print(f"Saved combined prediction to: {combined_output_path}")
        else:
            print(f"No objects detected in {image_path}")
    else:
        print(f"Failed to process image: {image_path}")

# Load your trained model
model = YOLO('./runs/detect/train3/weights/best.pt')

# Get the model's default input size
default_input_size = model.model.args['imgsz']
print(f"Default input size: {default_input_size}")

# Set up input and output folders
input_folder = './newImage/'
output_folder = './predictions/'
combined_folder = './pred_combined/'

# Create output folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(combined_folder, exist_ok=True)

# Process all PNG files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        process_and_save_image(model, image_path, output_folder, combined_folder)

print("Processing complete. All predictions saved in the 'predictions' and 'pred_combined' folders.")