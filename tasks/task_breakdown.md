# Task Breakdown: Satellite-Based Wildfire Detection and Monitoring

This document breaks down the project into more detailed tasks, corresponding to the phases outlined in the `project_timeline.md`.

## Phase 1: Foundation and Data Acquisition

*   **1.1 Project Setup & Planning:**

    *   [/] 1.1.1: Finalize detailed research questions and hypotheses. (MVP question defined)
    *   [/] 1.1.2: Conduct comprehensive literature review on SOTA methods and datasets. (Targeted review for model selection done)
    *   [X] 1.1.3: Establish Git repository with appropriate branching strategy.
    *   [X] 1.1.4: Set up Python virtual environment(s) with base dependencies (`pyproject.toml`).
    *   [X] 1.1.5: Plan cloud storage solution for raw and processed data. (Decision: Use Google Cloud Storage linked to GEE project)
    *   [ ] 1.1.6: Define coding standards and documentation guidelines. (Deferred for MVP)

*   **1.2 Study Area and Case Study Selection:**

    *   [X] 1.2.1: Identify 2-3 primary study regions based on wildfire frequency, data availability, and geographical diversity. (Decision: California)
    *   [X] 1.2.2: Select specific historical wildfire events within these regions for detailed case studies. (Decision: Dixie Fire 2021 for MVP)

*   **1.3 Data Acquisition Script Development (Earth Engine & APIs):**

    *   [X] 1.3.0: **Review and Prioritize Additional Datasets:** Based on XAI research documents, investigate and prioritize integration of GFBS, Google FireSat, NOAA HMS Smoke. (Decision: Deferred all for MVP)

    *   [ ] 1.3.1: **Sentinel-2:** (Next Step: Week 2)
        *   [ ] Develop EE scripts to search, filter (date, ROI, cloud cover), and select Sentinel-2 L2A imagery.
        *   [ ] Implement functions to select specific bands (e.g., RGB, NIR, SWIR).
        *   [ ] Implement functions to calculate spectral indices (NDVI, NBR, etc.) on EE.
        *   [ ] Develop EE scripts for exporting Sentinel-2 imagery/mosaics.
    *   [ ] 1.3.2: **FIRMS (Active Fires):** (Next Step: Week 2)
        *   [ ] Develop EE scripts or use API clients to query and download FIRMS data (MODIS & VIIRS).
        *   [ ] Implement filtering by confidence level and date/time.
        *   [ ] Handle conversion of FIRMS data to vector format (points).
    *   [X] 1.3.3: **ERA5 (Weather Data):** (Deferred for MVP)
        *   [ ] Develop EE scripts to access ERA5 daily/hourly data (temperature, precipitation, wind, humidity).
        *   [ ] Implement selection of relevant weather variables.
        *   [ ] Handle temporal aggregation if necessary (e.g., daily averages).
    *   [X] 1.3.4: **GFBS (Global Forest Burn Severity - Investigation):** (Deferred for MVP)
        *   [ ] Investigate access methods and data format.
        *   [ ] Assess suitability for training/validation of burn severity models.
    *   [X] 1.3.5: **Google FireSat (Investigation):** (Deferred for MVP)
        *   [ ] Explore API/EE access, data characteristics, and licensing.
        *   [ ] Evaluate for near real-time detection potential.
    *   [X] 1.3.6: **NOAA HMS Smoke Product (Investigation):** (Deferred for MVP)
        *   [ ] Check data access (shapefile, KML, WMS) and update frequency.
        *   [ ] Consider for contextual information or smoke segmentation task.

*   **1.4 Exploratory Data Analysis (EDA):**

    *   [ ] 1.4.1: Download sample datasets for each data source for a selected case study.
    *   [ ] 1.4.2: Perform initial quality checks (inspect for clouds, sensor errors, missing data).
    *   [ ] 1.4.3: Visualize sample imagery and data layers.
    *   [ ] 1.4.4: Analyze statistical distributions of pixel values and weather parameters.

*   **1.5 Initial Data Preprocessing Module Development:**

    *   [ ] 1.5.1: **Cloud Masking:** Implement or adapt existing algorithms for cloud and cloud shadow masking for Sentinel-2 (if not relying solely on L2A quality flags).
    *   [ ] 1.5.2: **Reprojection & Resampling:** Develop functions to reproject all data to a common CRS (e.g., UTM) and resample to a target resolution (e.g., 20m for Sentinel-2 bands).
    *   [ ] 1.5.3: **Normalization/Scaling:** Plan strategies for normalizing pixel values if required for ML models.

## Phase 2: Data Integration and Preprocessing Pipeline

*   **2.1 Full Preprocessing Pipeline Implementation:**

    *   [ ] 2.1.1: Integrate individual data acquisition and preprocessing scripts into a cohesive pipeline.
    *   [ ] 2.1.2: Automate the download or direct EE processing workflow.
    *   [ ] 2.1.3: Implement robust error handling and logging for the pipeline.

*   **2.2 Spatio-temporal Data Alignment:**

    *   [ ] 2.2.1: Ensure all raster data layers are aligned to the same grid (pixel resolution and extent).
    *   [ ] 2.2.2: Develop methods to align time-series data (e.g., daily weather with less frequent satellite passes).

*   **2.3 Analysis-Ready Data (ARD) Generation:**

    *   [ ] 2.3.1: Define the structure of ARD (e.g., multi-band raster stacks, data cubes).
    *   [ ] 2.3.2: Implement functions to create image patches (e.g., 256x256 pixels) for ML model input.
    *   [ ] 2.3.3: Pair image patches with corresponding labels (e.g., burned/unburned masks, active fire locations).

*   **2.4 Data Management and Versioning:**

    *   [ ] 2.4.1: Implement a clear naming convention and directory structure for raw, processed, and ARD.
    *   [ ] 2.4.2: Utilize DVC (Data Version Control) or similar tools for tracking datasets and models.

*   **2.5 Dataset Generation and QA/QC:**

    *   [ ] 2.5.1: Generate the first full version of the integrated dataset for selected case studies.
    *   [ ] 2.5.2: Perform thorough quality checks: visual inspection, statistical summaries, consistency checks.
    *   [ ] 2.5.3: Document dataset characteristics, known issues, and processing steps.

## Phase 3: Machine Learning Model Development, XAI, and Training

*   **3.1 ML Model Research and Selection:**

    *   [/] 3.1.1: Finalize selection of ML architectures for: (Decision: U-Net for Burned Area Mapping)
        *   Burned Area Mapping (e.g., U-Net, DeepLabv3+).
        *   Active Fire Detection (e.g., CNN-based classifiers, object detection models).

    *   [ ] 3.1.2: Research appropriate loss functions and evaluation metrics for each task.
    *   [X] 3.1.3: Research lightweight model architectures and optimization techniques (pruning, quantization, knowledge distillation) suitable for wildfire detection models. (Deferred for MVP)
    *   [/] 3.1.4: Research efficient XAI methods (e.g., SHAP approximations, LIME, rule extraction, saliency maps) applicable to selected model types. (Decision: Grad-CAM for MVP)

*   **3.2 Data Loading and Augmentation for ML:**

    *   [ ] 3.2.1: Develop PyTorch/TensorFlow `Dataset` and `DataLoader` classes for efficient data feeding.
    *   [ ] 3.2.2: Implement data augmentation techniques relevant to satellite imagery (e.g., rotations, flips, brightness adjustments).

*   **3.3 Baseline Model Implementation:**

    *   [ ] 3.3.1: Implement a simple baseline model for burned area mapping (e.g., a smaller U-Net).
    *   [ ] 3.3.2: Implement a simple baseline model for active fire detection.

*   **3.4 Model Training, Experimentation, and XAI Integration:**

    *   [ ] 3.4.1: Set up training scripts with experiment tracking (e.g., MLflow, Weights & Biases).
    *   [ ] 3.4.2: Train baseline models on the initial dataset.
    *   [ ] 3.4.3: Perform hyperparameter tuning for baseline models.
    *   [ ] 3.4.4: Experiment with more complex architectures and pre-trained backbones.
    *   [ ] 3.4.5: Investigate transfer learning from existing remote sensing models if applicable.
    *   [ ] 3.4.6: Implement selected lightweight model techniques.
    *   [ ] 3.4.7: Implement selected XAI methods and integrate with model training/inference workflow.
    *   [ ] 3.4.8: **Vertex AI Integration:**
        *   [ ] Set up Vertex AI project and necessary APIs.
        *   [ ] Configure Vertex AI Training jobs for model experiments (optional).
        *   [ ] Learn to deploy models to Vertex AI Endpoints.
        *   [ ] Configure `ExplanationSpec` for deployed models on Vertex AI to enable XAI features.

*   **3.5 Addressing Challenges:**

    *   [ ] 3.5.1: If class imbalance is an issue (e.g., few fire pixels), implement strategies like weighted loss, over/undersampling.
    *   [ ] 3.5.2: Monitor training for overfitting and apply regularization techniques.

*   **3.6 Evaluation Protocol Development:**

    *   [ ] 3.6.1: Define a clear train/validation/test split strategy (consider spatial and temporal splits).
    *   [ ] 3.6.2: Implement code to calculate relevant metrics (e.g., IoU, F1-score, precision, recall, accuracy for segmentation; AUC, F1 for classification).

## Phase 4: Evaluation, Refinement, XAI Integration, and Interactive Tools

*   **4.1 Rigorous Model Evaluation:**

    *   [ ] 4.1.1: Evaluate refined models on the hold-out test set(s).
    *   [ ] 4.1.2: Perform cross-case study evaluation to assess generalization.
    *   [ ] 4.1.3: Analyze model performance across different land cover types, fire sizes, and seasons.

*   **4.2 Error Analysis and Model Refinement:**

    *   [ ] 4.2.1: Visualize model predictions and identify common failure modes.
    *   [ ] 4.2.2: Based on error analysis, iterate on model architecture, data augmentation, or input features.
    *   [ ] 4.2.3: Evaluate the quality and consistency of XAI explanations (e.g., faithfulness, plausibility).
    *   [ ] 4.2.4: Refine XAI method parameters or choice based on evaluation.

*   **4.3 Jupyter Notebook Development for Visualization:**

    *   [ ] 4.3.1: Create notebooks to visualize input satellite imagery (RGB, false color, indices).
    *   [ ] 4.3.2: Develop notebooks to overlay FIRMS data and weather parameters on maps.
    *   [ ] 4.3.3: Create notebooks to display model predictions (e.g., burned area masks, heatmaps of fire probability) alongside ground truth.
    *   [ ] 4.3.4: Implement interactive elements (e.g., sliders for time, dropdowns for layers) using `ipywidgets` or `geemap` tools.
    *   [ ] 4.3.5: Create notebooks to demonstrate XAI outputs (saliency maps, feature attributions) from Vertex AI or local implementations, explaining model decisions for specific wildfire instances.

*   **4.4 Documentation:**
    *   [ ] 4.4.1: Document all data processing steps and parameters.
    *   [ ] 4.4.2: Document ML model architectures, training procedures, and hyperparameters.
    *   [ ] 4.4.3: Add clear explanations and usage instructions to Jupyter notebooks.
    *   [ ] 4.4.4: Document XAI methodologies, interpretation of results, and limitations.

## Phase 5: Finalization and Reporting

*   **5.1 Code and Notebook Finalization:**

    *   [ ] 5.1.1: Clean up and refactor all Python scripts and modules.
    *   [ ] 5.1.2: Ensure all Jupyter notebooks run correctly from start to finish and are well-commented.
    *   [ ] 5.1.3: Finalize `README.md` files for different parts of the project.

*   **5.2 End-to-End System Testing:**

    *   [ ] 5.2.1: Test the entire workflow from data acquisition to model prediction for a new, unseen wildfire event (if possible).

*   **5.3 Final Report Preparation:**

    *   [ ] 5.3.1: Draft introduction, background, and literature review sections.
    *   [ ] 5.3.2: Describe data sources and preprocessing methodology in detail.
    *   [ ] 5.3.3: Explain ML model architectures and training process.
    *   [ ] 5.3.4: Present and discuss model evaluation results, including quantitative metrics and qualitative examples.
    *   [ ] 5.3.5: Discuss limitations of the study and potential sources of error.
    *   [ ] 5.3.6: Outline conclusions and suggest directions for future research.

*   **5.4 Presentation and Archiving:**

    *   [ ] 5.4.1: Prepare slides for project presentation (if required).
    *   [ ] 5.4.2: Organize and archive all project code, data, models, and documentation in a structured manner.

## Phase 6: Edge Deployment Pilot (Optional Future Work)

*   **6.1 Edge Hardware and Software Setup:**
    *   [ ] 6.1.1: Research and procure suitable edge device(s) (e.g., Jetson Nano, Raspberry Pi 4/5, Coral Dev Board).
    *   [ ] 6.1.2: Set up OS, dependencies, and ML runtimes (e.g., TensorFlow Lite, ONNX Runtime) on the edge device.
*   **6.2 Model Conversion and Optimization for Edge:**
    *   [ ] 6.2.1: Convert trained lightweight models to an edge-compatible format (e.g., `.tflite`, `.onnx`).
    *   [ ] 6.2.2: Apply further post-training quantization or compilation if needed for the target hardware.
*   **6.3 Edge Inference Application Development:**
    *   [ ] 6.3.1: Develop a simple application/script on the edge device to load the model and perform inference on sample data (e.g., image patches).
    *   [ ] 6.3.2: Implement basic data input and prediction output mechanisms.
*   **6.4 Performance Evaluation on Edge:**
    *   [ ] 6.4.1: Benchmark inference speed (latency) and resource usage (CPU, memory, power) on the edge device.
    *   [ ] 6.4.2: Compare edge performance with server-based inference.
*   **6.5 Documentation and Reporting for Edge Pilot:**
    *   [ ] 6.5.1: Document the edge deployment process, challenges encountered, and performance results.
    *   [ ] 6.5.2: Summarize findings and provide recommendations for future operational edge deployment.