# Project Report: Wildfire Burn Area Segmentation

*Date: 2025-06-25*

---

## 1. End-to-End Workflow

The project follows a standard deep-learning pipeline:

1. **Data Ingestion & Preprocessing** – Raw Sentinel-2 satellite mosaics and FIRMS active-fire point data are downloaded from Google Cloud Storage. The point data are rasterised into a pixel-wise ground-truth *burn mask*.
2. **Patch Generation** – The large satellite image and corresponding burn mask are sliced into 256 × 256-pixel patches suitable for model training.
3. **Data Loading** – A custom PyTorch `Dataset` class loads the image-mask patches, normalises the input bands to `[0, 1]`, and converts everything to tensors.
4. **Model Training** – A U-Net model is trained on the patches. The training script handles the validation loop, loss computation, optimisation, and checkpointing of the best-performing weights.
5. **Evaluation** – Model performance is measured with the Dice similarity coefficient, a standard metric for image-segmentation tasks.

---

## 2. Component Analysis

### a) Data Preprocessing (`preprocessing.py`)

• **Source Data** – Sentinel-2 L2A Mosaic (`sentinel2_mosaic.tif`, five spectral bands) and FIRMS active-fire CSV (`firms_points_v2.csv`).

• **Key Steps**
  1. `download_data_from_gcs()` downloads both datasets from the GCS bucket.
  2. `create_burn_mask()` buffers each FIRMS point by 30 m and rasterises the result to create `burn_mask.tif` aligned with the Sentinel mosaic.
  3. `generate_training_patches()` iterates over the mosaic in 256 × 256 windows, saving only patches where at least 1 % of pixels are burned. This avoids an overwhelming majority of negative examples.

### b) PyTorch Dataset (`dataset.py`)

• **`BurnAreaDataset` Class** – Loads paired image and mask tiles.

• **Functionality** – Reads the `.tif` files, clips and scales Sentinel-2 values to `[0, 1]`, converts them to PyTorch tensors, and returns them as `(image, mask)` pairs. A `transform` hook exists for future data augmentation but is not yet used.

### c) U-Net Model Architecture (`unet_model.py`)

• **Implementation** – A classic U-Net, widely used for biomedical and satellite segmentation.

• **Structure** –
  • *Encoder* (contracting path) captures context while increasing feature depth from the 5-band input to 1024 channels.
  • *Decoder* (expansive path) upsamples the feature maps and concatenates them with high-resolution encoder features via skip connections.
  • A final 1 × 1 convolution outputs a single-channel burn mask.

### d) Training Script (`train.py`)

• **Framework** – Orchestrates the entire training and validation workflow.

• **Configuration** –
  • Device: CUDA if available, else CPU.
  • Hyper-parameters: batch size 8, 25 epochs, learning rate 1 × 10⁻⁴.

• **Process** –
  1. Instantiates `BurnAreaDataset` and splits it 85 %/15 % into training and validation sets.
  2. Creates `DataLoader`s for efficient batching.
  3. Initialises the U-Net, the Adam optimiser, and `BCEWithLogitsLoss`.
  4. Runs 25 training epochs, evaluating on the validation set each epoch.
  5. Saves `best_model.pth` whenever the validation Dice score improves.

### e) Evaluation Metric (`utils.py`)

`check_accuracy()` computes the Dice score by comparing thresholded model predictions with the ground-truth mask, giving a robust metric even under class imbalance.

---

## 3. Current Status & Recommendations

The project is in an excellent state: the end-to-end pipeline is complete and producing a baseline model. Code quality aligns with deep-learning best practices.

**Potential Next Steps**

1. **Implement Data Augmentation** – Add random flips, rotations, etc., within `BurnAreaDataset` to improve generalisation.
2. **Build an Inference Script** – Create `predict.py` that loads `best_model.pth`, tiles a large Sentinel-2 mosaic, and stitches the predicted burn mask.
3. **Enhance Logging** – Integrate TensorBoard or Weights & Biases to visualise training loss and Dice over time.
4. **Hyper-parameter Tuning** – Experiment with learning-rate schedules, optimisers, and batch sizes to squeeze more performance.

---

### Conclusion

All deliverables for Phase 1 (data preparation and baseline model) are finished. The next focus should be robustness (augmentation), usability (inference), and observability (logging).
