# main.py

import logging
import os
from data_processing import load_and_preprocess_data, prepare_data_for_yolo
from yolo_trainer import create_data_yaml, train_yolo_model, validate_yolo_model
from yolo_inference import perform_yolo_inference
from utils import setup_logging

def main():
    setup_logging()
    logging.info("Starting pipeline")

    # Define directories
    image_dir = "./images"
    mask_dir = "./masks"
    output_dir = "./output"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Ask user whether to train or perform inference
        mode = input("Enter 'train' to train a new model or 'infer' to run inference on a new image: ").lower()

        if mode == 'train':
            # Load and preprocess data
            images, masks = load_and_preprocess_data(image_dir, mask_dir)
            logging.info(f"Loaded and preprocessed {len(images)} image-mask pairs")

            # Prepare data for YOLOv8
            yolo_data = prepare_data_for_yolo(images, masks, output_dir)
            logging.info("Prepared data for YOLOv8 training")

            # Create data.yaml for YOLO training
            data_yaml_path = os.path.join(output_dir, "data.yaml")
            create_data_yaml(
                train_path=os.path.join(output_dir, "images"),
                val_path=os.path.join(output_dir, "images"),  # Using same data for validation
                nc=1,  # Number of classes (1 for airway)
                names=['airway'],
                yaml_path=data_yaml_path
            )

            # Train YOLOv8 model
            yolo_model = train_yolo_model(data_yaml_path, epochs=400)
            logging.info("YOLOv8 model training completed")

            # Validate YOLOv8 model
            validation_results = validate_yolo_model(yolo_model, data_yaml_path)
            logging.info(f"YOLOv8 validation completed. mAP50: {validation_results.map50}")

        elif mode == 'infer':
            perform_yolo_inference()

        else:
            logging.error("Invalid mode. Please enter 'train' or 'infer'.")

        logging.info("Pipeline completed successfully")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()