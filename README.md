# **Wildfire Burn Area Segmentation using U-Net**

This project utilizes a U-Net deep learning model to automatically identify and segment wildfire burn areas from Sentinel-2 satellite imagery. The goal is to provide a reliable and efficient method for detecting and monitoring wildfires, ultimately aiding in disaster response and management.

The project encompasses a complete end-to-end workflow, including data ingestion and preprocessing, patch generation, data loading, model training, and evaluation. The U-Net model is trained on a dataset of satellite images and corresponding burn masks, with a focus on achieving high accuracy in segmenting burn areas.

This repository provides a well-structured and maintainable codebase, following best practices for deep learning projects. It serves as a starting point for new contributors and users, offering a clear overview of the project's objectives, methodology, and current status.

---

#### **1. End-to-End Workflow**

The project follows a standard deep learning pipeline:

1.  **Data Ingestion & Preprocessing**: Raw Sentinel-2 satellite mosaics and FIRMS active fire point data are downloaded from Google Cloud Storage. The point data is converted into a pixel-wise ground truth "burn mask."
2.  **Patch Generation**: The large satellite image and corresponding burn mask are sliced into smaller 256x256 pixel patches suitable for training.
3.  **Data Loading**: A custom PyTorch `Dataset` class loads these image and mask patches, normalizes them, and converts them into tensors.
4.  **Model Training**: A U-Net model is trained on the patches. The training script includes a validation loop, loss calculation, optimization, and checkpointing to save the best-performing model.
5.  **Evaluation**: The model's performance is measured using the Dice similarity coefficient, a standard metric for image segmentation tasks.

---

#### **2. Component Analysis**

**a) Data Preprocessing (`preprocessing.py`)**

-   **Source Data**:
    -   Sentinel-2 L2A Mosaic (`sentinel2_mosaic.tif`): A 5-band multispectral image.
    -   FIRMS Active Fire Data (`firms_points_v2.csv`): A CSV file containing the coordinates of detected fire hotspots.
-   **Key Steps**:
    1.  `download_data_from_gcs()`: Fetches the raw data from a GCS bucket.
    2.  `create_burn_mask()`: This is a critical step that generates the ground truth labels. It reads the FIRMS fire point coordinates, applies a 30-meter buffer to each point to make them more significant at the image's resolution, and then "burns" these buffered shapes onto a blank raster, creating the `burn_mask.tif`.
    3.  `generate_training_patches()`: To create manageable inputs for the model, this function iterates over the large mosaic and mask in 256x256 windows. It implements a smart filtering mechanism, only saving a patch if the burn mask covers at least 1% of its area. This prevents the model from being overwhelmed by negative (non-burned) examples.

**b) PyTorch Dataset (`dataset.py`)**

-   **`BurnAreaDataset` Class**: This custom `Dataset` serves the preprocessed patches to the model.
-   **Functionality**:
    -   It reads corresponding image and mask `.tif` files.
    -   **Normalization**: Image pixel values (originally `uint16`) are clipped at 10,000 and scaled to a `[0, 1]` float range. This is a standard and effective normalization technique for Sentinel-2 data.
    -   The data is converted to PyTorch tensors for model consumption.
    -   A placeholder for data augmentation (`transform`) exists but is not currently used.

**c) U-Net Model Architecture (`unet_model.py`)**

-   **Implementation**: A classic U-Net architecture, which is the industry standard for biomedical and satellite image segmentation.
-   **Structure**:
    -   **Encoder (Contracting Path)**: Captures context by progressively downsampling the input image while increasing feature channels (from 5 input bands up to 1024 features).
    -   **Decoder (Expansive Path)**: Enables precise localization by gradually upsampling the feature maps and concatenating them with high-resolution features from the encoder path via "skip connections."
    -   **Output**: A final 1x1 convolution maps the features to a single-channel output, representing the predicted burn mask.

**d) Training Script (`train.py`)**

-   **Framework**: The script orchestrates the entire training and validation process.
-   **Configuration**:
    -   **Device**: Automatically uses a `CUDA` GPU if available, otherwise falls back to `CPU`.
    -   **Hyperparameters**:
        -   Batch Size: 8
        -   Epochs: 25
        -   Learning Rate: `1e-4`
-   **Process**:
    1.  The `BurnAreaDataset` is instantiated and split into 85% for training and 15% for validation.
    2.  PyTorch `DataLoader`s are created to efficiently feed data to the model in batches.
    3.  The `UNet` model is initialized, along with `Adam` optimizer and `BCEWithLogitsLoss` (a numerically stable loss function suitable for binary segmentation).
    4.  The main loop iterates for 25 epochs, training the model and evaluating it against the validation set at the end of each epoch.
    5.  **Checkpointing**: The model's weights are saved to `best_model.pth` only when the validation Dice score improves, ensuring the final saved model is the best one produced during training.

**e) Evaluation Metric (`utils.py`)**

-   **`check_accuracy()`**: This function calculates the Dice score, which measures the overlap between the predicted mask and the ground truth mask. It is a more informative metric than pixel-wise accuracy for segmentation, especially when classes are imbalanced (i.e., more non-burned than burned pixels).

---

#### **3. Current Status & Recommendations**

The project is in an excellent state, with a complete, end-to-end training pipeline. The code is well-structured and follows best practices for deep learning projects.

**Potential Next Steps:**

1.  **Implement Data Augmentation**: To improve model robustness and prevent overfitting, introduce random augmentations (e.g., horizontal/vertical flips, rotations) in the `BurnAreaDataset`. This is often the single most effective way to boost performance.
2.  **Build an Inference Script**: Create a new script (`predict.py`) that loads the saved `best_model.pth` and uses it to generate a burn mask for a new, unseen satellite image. This is the final step to make the model operational.
3.  **Enhance Logging**: Integrate a tool like **TensorBoard** or **Weights & Biases** to log and visualize the loss and Dice score during training. This provides much better insight into the model's learning dynamics.
4.  **Hyperparameter Tuning**: Experiment with different learning rates or optimizers to see if training can be improved.