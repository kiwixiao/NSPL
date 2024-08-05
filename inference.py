import torch
import numpy as np
from utils.io_utils import load_dataset
from utils.visualization import plot_segmentation_results

def infer(model, data_path, output_path, device):
    model.eval()
    dataset = load_dataset(data_path)
    
    with torch.no_grad():
        for i, (image, true_mask) in enumerate(dataset):
            image = image.unsqueeze(0).to(device)
            output = model(image)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            
            image = image.squeeze().cpu().numpy()
            true_mask = true_mask.squeeze().cpu().numpy()
            
            plot_segmentation_results(image, true_mask, pred_mask, f"{output_path}/segmentation_result_{i}.png")
            
            # Save predicted mask
            np.save(f"{output_path}/pred_mask_{i}.npy", pred_mask)
    
    print("Inference completed")