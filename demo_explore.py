
#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import torch

def load_and_check_image(image_path):
    try:
        # Attempt to open the image
        with Image.open(image_path) as img:
            print(f"Image format: {img.format}")
            print(f"Image mode: {img.mode}")
            print(f"Image size: {img.size}")
            
            # Convert to RGB (3 channels)
            img_rgb = img.convert('RGB')
            # override the above color model
            
            # Convert to numpy array
            img_array = np.array(img_rgb)
            
            print(f"Array shape: {img_array.shape}")
            print(f"Array dtype: {img_array.dtype}")
            print(f"Array min and max values: {img_array.min()}, {img_array.max()}")
            
            return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Load your trained model
model = YOLO('./runs/detect/train3/weights/best.pt')

# Get the model's default input size
default_input_size = model.model.args['imgsz']
print(f"Default input size: {default_input_size}")


#%%
# Load and check the image
image_path = './newImage/OSAMRI065_107.png'
input_image = load_and_check_image(image_path)

if input_image is not None:
    # Run inference
    results = model(input_image)

    # Plot the results
    plt.figure(figsize=(10, 10))
    plt.imshow(input_image)  # Remove cmap='gray' to show in color if it's RGB
    plt.axis('off')

    # Process and plot bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf.item()
            
            # Create a Rectangle patch
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            
            # Add label
            plt.text(x1, y1, f'Airway: {conf:.2f}', color='red', fontsize=12, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0))

    plt.title('MRI Airway Detection Results')
    plt.tight_layout()
    plt.show()
else:
    print("Failed to load the image. Please check the file path and format.")