# yolo_trainer.py

import logging
import yaml
from ultralytics import YOLO

def create_data_yaml(train_path, val_path, nc, names, yaml_path):
    data = {
        'train': train_path,
        'val': val_path,
        'nc': nc,
        'names': names
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    logging.info(f"Created data.yaml at {yaml_path}")

def train_yolo_model(data_yaml_path, epochs=400, imgsz=640):
    logging.info("Starting YOLOv8 model training")
    
    # Load a model
    model = YOLO('yolov8m.yaml')  # build a new model from YAML
    
    try:
        # Train the model
        results = model.train(data=data_yaml_path, epochs=epochs, imgsz=imgsz)
        
        logging.info("YOLOv8 training completed successfully")
        logging.info(f"Best mAP50: {results.best_map50}")
        
        # Log the path where the model was automatically saved
        logging.info(f"Best model saved at: {results.best}")
        
        return model
    
    except Exception as e:
        logging.error(f"An error occurred during YOLO training: {str(e)}")
        raise

def validate_yolo_model(model, data_yaml_path):
    logging.info("Validating YOLOv8 model")
    try:
        results = model.val(data=data_yaml_path)
        logging.info(f"Validation mAP50: {results.map50}")
        return results
    except Exception as e:
        logging.error(f"An error occurred during YOLO validation: {str(e)}")
        raise