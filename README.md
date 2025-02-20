# Water Segmentation Using U-Net

This project is focused on segmenting water bodies in multispectral satellite images using a deep learning approach, specifically the U-Net architecture. The goal is to generate binary segmentation masks where water bodies are identified from the satellite images.

## Project Overview

### Purpose

The project involves the following key steps:
1. **Data Preprocessing**: Preparing the images and segmentation masks for training.
2. **Modeling**: Implementing a U-Net architecture for water segmentation.
3. **Training**: Training the U-Net model on the preprocessed data.
4. **Evaluation**: Assessing the performance of the model using metrics like IoU, Precision, Recall, and F1-Score.
5. **Visualization**: Visualizing results, including ground truth masks and model predictions with heatmaps.

### Project Structure

The project directory structure is as follows:
- **data/images**: Contains the multispectral and optical satellite images.
- **data/labels**: Contains the original binary segmentation masks corresponding to the images.
- **data/filtered_labels**: Contains the filtered and validated masks ready for training.
- **src/**: Includes all the scripts for data preprocessing, model definition, and training.

## Data

### Input Data

The input consists of multispectral satellite images, which are stored in the `data/images` directory. Each image contains multiple bands (13 in this case), representing different spectral bands from the satellite sensors, including NIR (Near Infrared) and SWIR (Short-Wave Infrared) bands. The corresponding segmentation masks, stored in the `data/labels` directory, indicate areas covered by water (1) and non-water (0).

### Label Filtering

To ensure that the model only uses relevant labels, the masks are filtered based on their filenames. Only valid masks that follow the pattern of being a simple integer (e.g., `5.png`) are retained and moved to the `data/filtered_labels` directory. This step helps avoid potential issues with incorrectly labeled data.

### Preprocessing

Once the valid labels are filtered and stored, the images and labels are loaded and preprocessed:
- The images are normalized to the range `[0, 1]`.
- The labels are converted into `uint8` format, ensuring compatibility with the model's expectations.
- **NDWI (Normalized Difference Water Index)** is computed from the NIR and SWIR bands and concatenated to the images for better water identification.

## Model

### U-Net Architecture

The core of the model is based on the **U-Net** architecture, which is widely used for image segmentation tasks. It consists of:
- **Downsampling (Encoder)**: Several convolutional layers followed by max-pooling operations to capture low-level features and reduce spatial dimensions.
- **Bottleneck**: A transition between the encoder and decoder that captures the most abstract representations.
- **Upsampling (Decoder)**: A series of convolutional transpose operations to restore the spatial dimensions, combined with skip connections from the encoder to retain detailed features.
- **Output Layer**: A final convolutional layer with a sigmoid activation to generate a binary mask indicating the presence of water.

### Metrics

The model is evaluated using various metrics, including:
- **Mean IoU (Intersection over Union)**: Measures the overlap between predicted and ground truth masks.
- **Precision**: The proportion of true positive water pixels among all predicted water pixels.
- **Recall**: The proportion of true positive water pixels among all actual water pixels in the ground truth.
- **F1-Score**: The harmonic mean of Precision and Recall, providing a balanced evaluation metric.

## Training Process

The model is trained on the preprocessed data with the following steps:
1. **Label Preparation**: The `prepare_labels()` function filters and moves valid labels from `data/labels` to `data/filtered_labels`.
2. **Data Loading and Preprocessing**: The satellite images and labels are loaded, NDWI is computed, and images are normalized and combined with the NDWI values.
3. **Model Compilation**: The U-Net model is instantiated and compiled with the Adam optimizer and binary cross-entropy loss function. The model is evaluated using the Mean IoU metric.
4. **Training**: The model is trained using a validation split and early stopping to avoid overfitting. The `ModelCheckpoint` callback is used to save the best model during training.

## Visualization

### Heatmap Visualization

To visualize the model's performance, the project includes a heatmap visualization function:
- **RGB Composite**: The first three bands of the image are displayed as an RGB composite.
- **NDWI Heatmap**: A heatmap is generated for the NDWI, indicating areas with high water content.
- **Ground Truth Mask**: The actual mask of water and non-water areas is shown.
- **Predicted Mask**: The predicted water segmentation mask is overlaid on the RGB image.
- **NIR and SWIR Bands**: The individual NIR and SWIR bands are displayed to show the contribution of these spectral bands to the segmentation task.

## Usage

1. **Prepare the Data**:
   - The `prepare_labels()` function filters and moves valid labels from `data/labels` to `data/filtered_labels`.
   - The images and labels are then preprocessed and prepared for training, including the calculation of NDWI.
   
2. **Train the Model**:
   - The U-Net model is instantiated, compiled, and trained using the preprocessed data.

3. **Evaluate the Model**:
   - The model's performance is evaluated using the metrics mentioned above, and the results are analyzed to assess how well the model performs in segmenting water bodies from satellite images.

4. **Visualize the Results**:
   - The heatmap visualization function can be used to compare the ground truth masks with the predicted segmentation masks, providing insights into the model's accuracy.

## Conclusion

This project demonstrates how the U-Net architecture can be effectively used for water segmentation tasks on satellite images. By properly preparing the data, including calculating NDWI, and utilizing a deep learning model tailored for segmentation, we can accurately identify water bodies from multispectral satellite imagery.
