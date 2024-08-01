# yolo_inference.py

import os
import logging
import glob
from ultralytics import YOLO
import cv2
import numpy as np

def load_latest_yolo_model():
    # Find the latest run directory
    run_dirs = glob.glob('runs/detect/train*')
    if not run_dirs:
        raise FileNotFoundError("No trained YOLO model found. Please train a model first.")
    
    latest_run = max(run_dirs, key=os.path.getmtime)
    
    # Find the best.pt file in the latest run directory
    model_path = os.path.join(latest_run, 'weights', 'best.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No 'best.pt' file found in {latest_run}")
    
    logging.info(f"Loading latest YOLO model from: {model_path}")
    return YOLO(model_path)

def perform_yolo_inference():
    # Load the latest trained model
    model = load_latest_yolo_model()
    
    # Prompt for new image location
    image_path = input("Enter the path to the new image for inference (or press Enter to exit): ")
    
    if not image_path:
        raise ValueError("No image path provided. Exiting inference.")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The specified image path does not exist: {image_path}")
    
    # Perform inference
    logging.info(f"Performing inference on image: {image_path}")
    results = model(image_path)
    
    # Process and display results
    for result in results:
        img = cv2.imread(image_path)
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"Class: {cls}, Conf: {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the image with bounding boxes
        cv2.imshow("Inference Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    logging.info("Inference completed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        perform_yolo_inference()
    except Exception as e:
        logging.error(f"An error occurred during inference: {str(e)}")