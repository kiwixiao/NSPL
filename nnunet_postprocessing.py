import numpy as np
from scipy import ndimage

def remove_small_connected_components(segmentation, min_size=100):
    labeled_seg, num_features = ndimage.label(segmentation)
    component_sizes = np.bincount(labeled_seg.ravel())
    too_small = component_sizes < min_size
    too_small_mask = too_small[labeled_seg]
    segmentation[too_small_mask] = 0
    return segmentation

def postprocess_prediction(prediction, threshold=0.5):
    binary_prediction = (prediction > threshold).astype(np.uint8)
    postprocessed = remove_small_connected_components(binary_prediction)
    return postprocessed