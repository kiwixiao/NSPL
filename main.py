# main.py


import os
import logging
import argparse
from data_processing import load_and_preprocess_data, prepare_data_for_yolo
from yolo_trainer import train_yolo_model, validate_yolo_model
from unet_segmentation import train_unet, evaluate_unet
from unet_infer import perform_unet_inference
from utils import setup_logging, create_output_dirs, save_config, load_config

def main():
    parser = argparse.ArgumentParser(description="Airway Segmentation Pipeline")
    parser.add_argument('--mode', choices=['train_yolo', 'train_unet', 'infer'], required=True,
                        help="Mode of operation: train_yolo, train_unet, or infer")
    parser.add_argument('--image_dir', help="Directory containing MRI images")
    parser.add_argument('--mask_dir', help="Directory containing mask images")
    parser.add_argument('--output_dir', default='output', help="Directory to save output files")
    parser.add_argument('--yolo_model', help="Path to YOLO model for inference")
    parser.add_argument('--unet_model', help="Path to U-Net model for inference")
    parser.add_argument('--infer_image', help="Path to image for inference")
    parser.add_argument('--config', help="Path to configuration file")
    args = parser.parse_args()

    # Setup logging
    setup_logging()
    
    # Create output directories
    dirs = create_output_dirs(args.output_dir)
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = vars(args)
        save_config(config, os.path.join(args.output_dir, 'config.txt'))

    logging.info(f"Starting pipeline in {args.mode} mode")

    try:
        if args.mode == 'train_yolo':
            if not config['image_dir'] or not config['mask_dir']:
                raise ValueError("Image and mask directories are required for YOLO training")

            # Load and preprocess data
            images, masks = load_and_preprocess_data(config['image_dir'], config['mask_dir'])
            logging.info(f"Loaded and preprocessed {len(images)} image-mask pairs")

            # Prepare data for YOLOv8
            yolo_data = prepare_data_for_yolo(images, masks, dirs['yolo_data'])
            logging.info("Prepared data for YOLOv8 training")

            # Train YOLOv8 model
            yolo_model = train_yolo_model(yolo_data, epochs=100, output_dir=dirs['yolo_output'])
            logging.info("YOLOv8 model training completed")

            # Validate YOLOv8 model
            validate_yolo_model(yolo_model, yolo_data)

        elif args.mode == 'train_unet':
            if not config['image_dir'] or not config['mask_dir']:
                raise ValueError("Image and mask directories are required for U-Net training")

            # Train U-Net model
            train_unet(config['image_dir'], config['mask_dir'], dirs['unet_output'])
            logging.info("U-Net model training completed")

            # Evaluate U-Net model
            evaluate_unet(dirs['unet_output'])

        elif args.mode == 'infer':
            if not config['infer_image']:
                raise ValueError("Image path is required for inference")
            if not config['yolo_model'] or not config['unet_model']:
                raise ValueError("Both YOLO and U-Net model paths are required for inference")

            # Perform inference
            perform_unet_inference(config['infer_image'], config['yolo_model'], config['unet_model'], dirs['inference_output'])

        logging.info("Pipeline completed successfully")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    
    """
    # YOLO training
    python main.py --mode train_yolo --image_dir /path/to/images --mask_dir /path/to/masks --output_dir /path/to/output
    # U-net training
    python main.py --mode train_unet --image_dir /path/to/cropped_images --mask_dir /path/to/cropped_masks --output_dir /path/to/output
    
    # For inference
    python main.py --mode infer --infer_image /path/to/image.nii.gz --yolo_model /path/to/yolo_model.pt --unet_model /path/to/unet_model.pth --output_dir /path/to/output
    
    """
    main()