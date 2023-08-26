import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Define the path to the dataset
dataset_path = "."

def normalize_image(image_data):
    return (image_data - image_data.min()) / (image_data.max() - image_data.min())

def load_data(files, data_type="imagesTr"):
    print("Loading data...")
    data = [normalize_image(nib.load(os.path.join(dataset_path, data_type, file)).get_fdata()) for file in files]
    data = np.array(data)
    data = np.mean(data, axis=-1)
    print(f"Loaded {len(data)} {data_type}.")
    return data.reshape(data.shape + (1,))

def create_unet_model():
    print("Creating U-Net model...")
    inputs = Input((240, 240, 155, 1))
    
    # Encoder
    c1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling3D((2, 2, 2))(c2)
    
    # Bottleneck
    c3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(p2)
    
    # Decoder
    u2 = UpSampling3D((2, 2, 2))(c3)
    c4 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(u2)
    u1 = UpSampling3D((2, 2, 2))(c4)
    c5 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(u1)
    
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(c5)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("U-Net model created.")
    return model

if __name__ == "__main__":
    print("Starting the program...")
    
    image_files = [f for f in os.listdir(os.path.join(dataset_path, "imagesTr")) if not f.startswith("._")]
    X_data = load_data(image_files, data_type="imagesTr")
    Y_data = load_data(image_files, data_type="labelsTr")
    
    # Split data into training and validation sets
    print("Splitting data into training and validation sets...")
    X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
    
    model = create_unet_model()
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1)
    
    print("Starting training...")
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50, batch_size=2, callbacks=[early_stopping, model_checkpoint])
    print("Training completed.")
    
    with open('training_history.pkl', 'wb') as f:
        print("Saving training history...")
        pickle.dump(history.history, f)
    
    # Load best model
    print("Loading best model...")
    model.load_weights("best_model.h5")
    
    # Evaluate on a sample
    print("Evaluating on a sample...")
    sample_original = X_val[:10]
    sample_segmentation = model.predict(sample_original)
    
    # Save results
    print("Saving results...")
    np.save('sample_original.npy', sample_original)
    np.save('sample_segmentation.npy', sample_segmentation)
    
    # Visualization
    print("Visualizing results...")
    plt.imshow(sample_original[0, :, :, 77, 0], cmap='gray')
    plt.imshow(sample_segmentation[0, :, :, 77, 0], alpha=0.5, cmap='jet')
    plt.title("Original Image with Segmentation Overlay")
    plt.axis('off')
    plt.savefig("segmentation_overlay.png")  # Save the plot
    plt.show()
    print("Visualization completed.")

