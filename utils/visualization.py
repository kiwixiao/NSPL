import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_segmentation_results(image, true_mask, pred_mask, save_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(true_mask, cmap='jet')
    ax2.set_title('True Mask')
    ax2.axis('off')
    
    ax3.imshow(pred_mask, cmap='jet')
    ax3.set_title('Predicted Mask')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()