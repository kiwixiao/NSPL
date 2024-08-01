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
            # ... (training code remains the same)
            pass
        elif mode == 'infer':
            # Ask if the user wants to use a specific model
            use_specific_model = input("Do you want to use a specific model? (y/n): ").lower() == 'y'
            if use_specific_model:
                model_path = input("Enter the path to the model file: ")
                perform_yolo_inference(model_path)
            else:
                perform_yolo_inference()
        else:
            logging.error("Invalid mode. Please enter 'train' or 'infer'.")

        logging.info("Pipeline completed successfully")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()