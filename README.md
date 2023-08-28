# Brain Tumor Segmentation

This repository contains code for segmenting brain tumors from MRI scans.

## Directory Structure:

- `model/`: Contains the trained model `best_model.h5`.
- `outputs/`: Contains output visualizations and results.
- `scripts/`: Contains the main Python scripts for data processing, model training, and evaluation.
- `bash_scripts/`: Contains bash scripts for running the Python scripts.

## Dataset Information

The dataset we are using is from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/), specifically the Brain Tumours dataset.

The dataset consists of MRI scans in the `.nii.gz` format, which is a standard format for storing medical imaging data. Our goal is to preprocess this data, build and train an autoencoder, and then use the trained model for segmentation tasks.

### Dataset Structure:

- **imagesTr**: This directory contains the training images. These are the MRI scans that you will use to train your models.
- **imagesTs**: This directory contains the test images. These are the MRI scans that you can use to evaluate the performance of your models after training.
- **labelsTr**: This directory contains the ground truth labels (segmentations) corresponding to the training images in imagesTr. These labels indicate the regions in the MRI scans where tumors are present. They will be used as the "target" when training segmentation models.
- **dataset.json**: This file likely contains metadata about the dataset, such as details about the image acquisition, annotations, and other relevant information.


## Steps:

1. **Brain Segmentation**: 
   - Script: `scripts/Brain_Segmentation.py`
   - Bash Script: `bash_scripts/run_Brain_Segmentation.sh`
   - Description: This script is responsible for training a U-Net model on brain MRI data for segmentation tasks. It loads the MRI data, defines the U-Net model, trains it, and saves the results. Key functions include `normalize_image()`, `load_data()`, and `create_unet_model()`.

2. **U-Net Evaluation**: 
   - Script: `scripts/unet_evaluation.py`
   - Bash Script: `bash_scripts/run_unet_evaluation.sh`
   - Description: Evaluates the performance of the trained U-Net model on validation data. It loads the trained model, generates predictions for the validation set, and computes metrics like Dice Coefficient and IoU.

3. **Analysis and Visualization**: 
   - Script: `scripts/AnalysisandVisualization.py`
   - Description: This script provides visual insights into the model's learning process. It overlays the predicted segmentations on the original images, computes error maps, and plots training curves to understand the model's learning trajectory.

4. **Data Augmentation**: 
   - Script: `scripts/augmantation.py`
   - Description: Augments the MRI data to create more diverse training samples. It defines a sequence of augmentations and applies them to the MRI data.

5. **Augmentation Comparison**: 
   - Script: `scripts/augmantation_comparison.py`
   - Description: Compares the original MRI data with the augmented data. It visualizes side-by-side comparisons of original and augmented images and masks.

## Issues to Address:
These are my next steps
- If the model isn't performing well, consider adjusting hyperparameters or checking the training data.
- Ensure that the segmentations make sense. If the masks are consistently blank, there might be an issue with the ground truth data or model predictions.
- Ensure that the augmentations are appropriate and not overly distorting the data.
- Ensure that visualizations are correctly overlaying segmentations on the original images.

## Future Work:

- Correct the issues mentioned above.
- Fine-tune the model for better performance.
- Incorporate more advanced augmentation techniques.

