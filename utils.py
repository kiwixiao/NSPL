# utils.py

import os
import logging
import datetime

def setup_logging(log_dir='logs'):
    """
    Set up logging configuration.
    
    Args:
    log_dir (str): Directory to store log files.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging setup complete. Log file: {log_filename}")

def create_output_dirs(base_dir):
    """
    Create necessary output directories.
    
    Args:
    base_dir (str): Base directory for outputs.
    
    Returns:
    dict: Dictionary containing paths to created directories.
    """
    dirs = {
        'yolo_data': os.path.join(base_dir, 'yolo_data'),
        'yolo_output': os.path.join(base_dir, 'yolo_output'),
        'unet_data': os.path.join(base_dir, 'unet_data'),
        'unet_output': os.path.join(base_dir, 'unet_output'),
        'inference_output': os.path.join(base_dir, 'inference_output')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")
    
    return dirs

def save_config(config, filename):
    """
    Save configuration to a file.
    
    Args:
    config (dict): Configuration dictionary.
    filename (str): Path to save the configuration file.
    """
    with open(filename, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    logging.info(f"Configuration saved to {filename}")

def load_config(filename):
    """
    Load configuration from a file.
    
    Args:
    filename (str): Path to the configuration file.
    
    Returns:
    dict: Loaded configuration.
    """
    config = {}
    with open(filename, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            config[key] = value
    logging.info(f"Configuration loaded from {filename}")
    return config

# You can add more utility functions here as needed