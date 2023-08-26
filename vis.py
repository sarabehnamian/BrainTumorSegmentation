import numpy as np
import matplotlib.pyplot as plt

# Load the data
X_data = np.load('sample_original.npy')
Y_data = np.load('sample_segmentation.npy')

# Visualize multiple slices and samples
for sample_idx in range(3):  # First 3 samples
    for slice_idx in range(70, 80):  # Slices 70 to 79
        plt.figure(figsize=(10, 5))
        
        # Original Image
        plt.subplot(1, 2, 1)
        plt.imshow(X_data[sample_idx, :, :, slice_idx, 0], cmap='gray')
        plt.title(f"Original Image - Sample {sample_idx+1}, Slice {slice_idx}")
        plt.axis('off')
        
        # Mask
        plt.subplot(1, 2, 2)
        plt.imshow(Y_data[sample_idx, :, :, slice_idx, 0], cmap='gray')
        plt.title(f"Mask - Sample {sample_idx+1}, Slice {slice_idx}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

