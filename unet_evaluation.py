import os
import numpy as np
import nibabel as nib
from sklearn.metrics import jaccard_score
from tensorflow.keras.models import load_model

# Define the path to the dataset
dataset_path = "."

def normalize_image(image_data):
    return (image_data - image_data.min()) / (image_data.max() - image_data.min())

def load_data(files, data_type="imagesTr"):
    data = [normalize_image(nib.load(os.path.join(dataset_path, data_type, file)).get_fdata()) for file in files]
    data = np.array(data)
    data = np.mean(data, axis=-1)
    return data.reshape(data.shape + (1,))

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

# Load the data
image_files = [f for f in os.listdir(os.path.join(dataset_path, "imagesTr")) if not f.startswith("._")]
X_data = load_data(image_files, data_type="imagesTr")
Y_data = load_data(image_files, data_type="labelsTr")

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

# Load the trained model
model = load_model("best_model.h5")

# Generate predictions for the validation set
sample_segmentation = model.predict(X_val)

# Compute Dice Coefficient and IoU
y_true_flat = Y_val.flatten()
y_pred_flat = (sample_segmentation > 0.5).astype(int).flatten()  # Threshold predictions

dice = dice_coefficient(y_true_flat, y_pred_flat)
iou = jaccard_score(y_true_flat, y_pred_flat)

# Save results to a file
with open("evaluation_results.txt", "w") as file:
    file.write(f"Dice Coefficient: {dice:.4f}\n")
    file.write(f"IoU (Jaccard Index): {iou:.4f}\n")

print(f"Dice Coefficient: {dice:.4f}")
print(f"IoU (Jaccard Index): {iou:.4f}")

