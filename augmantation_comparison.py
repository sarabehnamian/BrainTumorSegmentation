import numpy as np
import matplotlib.pyplot as plt

# Load the original and augmented data
X_data = np.load('sample_original.npy')
Y_data = np.load('sample_segmentation.npy')
X_augmented = np.load('X_augmented.npy')
Y_augmented = np.load('Y_augmented.npy')

# Choose a sample index and slice number for visualization
sample_idx = 0
slice_num = 77

plt.figure(figsize=(10, 8))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(X_data[sample_idx, :, :, slice_num, 0], cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Original Mask
plt.subplot(2, 2, 2)
plt.imshow(Y_data[sample_idx, :, :, slice_num, 0], cmap='gray')
plt.title("Original Mask")
plt.axis('off')

# Augmented Image
plt.subplot(2, 2, 3)
plt.imshow(X_augmented[sample_idx, :, :, slice_num], cmap='gray')
plt.title("Augmented Image")
plt.axis('off')

# Augmented Mask
plt.subplot(2, 2, 4)
plt.imshow(Y_augmented[sample_idx, :, :, slice_num], cmap='gray')
plt.title("Augmented Mask")
plt.axis('off')

plt.tight_layout()

# Save the visualization to a file
plt.savefig("augmentation_comparison.png")

# Optionally display the visualization
plt.show()
"""Let's interpret them:

Original Image: This is the raw medical image you provided. It should look like a typical brain MRI slice.

Original Mask: If it's completely black, it suggests that there might not be any labeled tumor or region of interest in that particular slice. Remember, not all slices will have tumors or the specific region you're trying to segment. It's also possible that there's an issue with the ground truth data or the way it was loaded.

Augmented Image: This is the transformed version of the original image. It might look "funny" because augmentation techniques like rotation, elastic transformations, etc., can distort the image to generate new training samples. The purpose is to make the model more robust by exposing it to various possible transformations of the input data.

Augmented Mask: If this is also black, it suggests that the corresponding augmented image slice also doesn't have any labeled region. This is consistent with the original mask being black. If the original mask doesn't have any labeled region, the augmented mask will also be empty after transformations."""

