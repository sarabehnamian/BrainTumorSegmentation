import numpy as np
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa

# Define the path to the dataset
dataset_path = "."

# ==========================
# Load the Data
# ==========================
X_data = np.load('sample_original.npy')
Y_data = np.load('sample_segmentation.npy')

# ==========================
# Diagnostic: Check Unique Values in Masks
# ==========================
unique_values = np.unique(Y_data)
print(f"Unique values in masks: {unique_values}")

# ==========================
# Verify Ground Truth
# ==========================
for i in range(5):  # Displaying 5 samples
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(X_data[i, :, :, 77, 0], cmap='gray')
    plt.title(f"Original Image - Sample {i+1}")
    plt.axis('off')
    
    # Original Mask
    plt.subplot(1, 2, 2)
    plt.imshow(Y_data[i, :, :, 77, 0], cmap='gray', vmin=0, vmax=1)  # Adjusted visualization
    plt.title(f"Original Mask - Sample {i+1}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"outputs/original_image_and_mask_sample_{i+1}.png")
    plt.close()

# ==========================
# Adjust Augmentation Parameters
# ==========================
seq = iaa.Sequential([
    iaa.Affine(rotate=(-10, 10)),  # rotate images by -10 to 10 degrees
    iaa.ElasticTransformation(alpha=20, sigma=2)  # apply milder elastic transformations
])

# ==========================
# Visualize More Slices
# ==========================
sample_idx = 0
for slice_idx in range(70, 80):  # Displaying slices 70 to 79
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(X_data[sample_idx, :, :, slice_idx, 0], cmap='gray')
    plt.title(f"Original Image - Slice {slice_idx}")
    plt.axis('off')
    
    # Original Mask
    plt.subplot(1, 2, 2)
    plt.imshow(Y_data[sample_idx, :, :, slice_idx, 0], cmap='gray', vmin=0, vmax=1)  # Adjusted visualization
    plt.title(f"Original Mask - Slice {slice_idx}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"outputs/original_image_and_mask_slice_{slice_idx}.png")
    plt.close()

# ==========================
# Check Data Range
# ==========================
if not np.array_equal(unique_values, [0, 1]):
    Y_data = np.where(Y_data > 0.5, 1, 0)
    print("Masks have been thresholded to binary values.")



