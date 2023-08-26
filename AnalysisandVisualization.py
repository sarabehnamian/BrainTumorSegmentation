import numpy as np
import matplotlib.pyplot as plt
import pickle

# ==========================
# 1. Visual Inspection
# ==========================

# Load the original images and their corresponding segmentations
sample_original = np.load('sample_original.npy')
sample_segmentation = np.load('sample_segmentation.npy')

# Overlay Predictions on Original Images:
# This step helps in visually assessing how well the predicted segmentations align with the original images.

# Overlay Predictions on Original Images:
for i in range(sample_original.shape[0]):
    plt.figure(figsize=(10, 5))
    
    # Displaying the original image with the predicted segmentation overlay
    plt.subplot(1, 2, 1)
    plt.imshow(sample_original[i, :, :, 77, 0], cmap='gray')
    plt.imshow(sample_segmentation[i, :, :, 77, 0], alpha=0.5, cmap='jet')
    plt.title("Original Image with Segmentation Overlay - Sample {}".format(i+1))
    plt.axis('off')
    
    # Error Maps:
    plt.subplot(1, 2, 2)
    error_map = np.abs(sample_original[i, :, :, 77, 0] - sample_segmentation[i, :, :, 77, 0])
    plt.imshow(error_map, cmap='hot')
    plt.title("Error Map - Sample {}".format(i+1))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("overlay_and_error_sample_{}.png".format(i+1))
    plt.show()

# ==========================
# 2. Model Analysis
# ==========================

# Load the training history to analyze the model's performance over epochs
with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Training Curves:
# These plots provide insights into the model's learning process.
# They can help diagnose issues like overfitting (where validation loss starts increasing while training loss decreases).
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

