import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np

# Define the path to the dataset
dataset_path = "."

# Load the data
X_data = np.load('sample_original.npy')
Y_data = np.load('sample_segmentation.npy')

# Define the augmentation sequence
seq = iaa.Sequential([
    iaa.Affine(rotate=(-15, 15)),  # rotate images by -15 to 15 degrees
    iaa.ElasticTransformation(alpha=30, sigma=3)  # apply elastic transformations
])

def augment_3d_volume(volume, augmenter):
    augmented_volume = np.zeros_like(volume)
    depth = volume.shape[2]
    for z in range(depth):
        slice_2d = volume[:, :, z]
        augmented_slice_2d = augmenter.augment_image(slice_2d)
        augmented_volume[:, :, z] = augmented_slice_2d
    return augmented_volume

# Apply the augmentations
X_augmented = np.array([augment_3d_volume(x, seq) for x in X_data])
Y_augmented_raw = np.array([augment_3d_volume(y, seq) for y in Y_data])

# Threshold the augmented masks to ensure they remain binary
Y_augmented = np.where(Y_augmented_raw > 0.5, 1, 0)

# Save the augmented data
np.save('X_augmented.npy', X_augmented)
np.save('Y_augmented.npy', Y_augmented)



