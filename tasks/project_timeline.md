# Project Timeline: Satellite-Based Wildfire Detection (MVP - 2 Months)

This timeline outlines an accelerated 2-month (approx. 8-week) plan to deliver a Minimum Viable Product (MVP). It assumes a start date around **June 2025** and aims for completion by **early August 2025**. This requires a highly focused scope.

## Overall MVP Goal:
Develop and evaluate a core satellite-based wildfire detection/mapping model (e.g., burned area) using 1-2 primary datasets (Sentinel-2, FIRMS), implement one basic XAI technique, and produce a concise report.

## Phase 1: Focused Setup & Core Data Acquisition (Weeks 1-2: June 2025)

*   **Week 1 (Early June):**
    *   [ ] Solidify MVP scope: Define one specific wildfire event/region for analysis.
    *   [ ] Confirm primary datasets: Sentinel-2 L2A and FIRMS active fire data for the selected event.
    *   [ ] Set up project environment: Git repository, Python virtual environment, core libraries (`rasterio`, `geopandas`, `tensorflow`/`pytorch`, `scikit-learn`).
    *   [ ] Basic literature check for a suitable pre-existing model architecture (e.g., U-Net) for burned area mapping or active fire detection.
*   **Week 2 (Mid June):**
    *   [ ] Acquire Sentinel-2 imagery for the chosen event (1-2 scenes).
    *   [ ] Acquire FIRMS data for the corresponding period and region.
    *   [ ] Perform initial Exploratory Data Analysis (EDA) on the acquired data.
    *   [ ] Implement basic preprocessing: Cloud masking for Sentinel-2 scenes, spatial alignment of FIRMS points to Sentinel-2 pixels.

## Phase 2: MVP Model Development & Basic XAI (Weeks 3-5: Late June - Mid July 2025)

*   **Week 3 (Late June):**
    *   [ ] Implement the chosen ML model architecture (e.g., U-Net) using TensorFlow/Keras or PyTorch.
    *   [ ] Develop data loaders to feed preprocessed data patches (image + label) to the model.
    *   [ ] Create binary masks for training (e.g., burned vs. unburned) from FIRMS or manual interpretation if necessary for the small scope.
*   **Week 4 (Early July):**
    *   [ ] Train the baseline model on the prepared dataset.
    *   [ ] Conduct initial evaluation: Visual inspection of predictions, basic metrics (e.g., IoU, F1-score on a small validation set).
    *   [ ] Iterate once on hyperparameters if initial results are poor and time permits.
*   **Week 5 (Mid July):**
    *   [ ] Implement one basic XAI technique applicable to the model (e.g., Grad-CAM for CNN-based models, or feature occlusion if simpler).
    *   [ ] Generate XAI visualizations (saliency maps) for a few correct and incorrect predictions.
    *   [ ] Briefly document the XAI method and initial observations.

## Phase 3: Focused Evaluation, Documentation & Reporting (Weeks 6-8: Mid July - Early August 2025)

*   **Week 6 (Late July):**
    *   [ ] Conduct a focused evaluation of the MVP model on a small, held-out test set from the selected event.
    *   [ ] Analyze key results, common failure modes, and limitations of the MVP.
    *   [ ] (Optional - if significant time remains & prior setup is smooth) Quick test: Attempt to package the model and run a prediction using Vertex AI SDK (no full deployment, just a prediction call).
*   **Week 7 (Early August):**
    *   [ ] Document the streamlined data pipeline, MVP model architecture, training process, evaluation results, and basic XAI findings in a concise report/Jupyter notebook.
    *   [ ] Create key visualizations: Input data, ground truth, model predictions, XAI maps.
*   **Week 8 (Early August):**
    *   [ ] Finalize the MVP project report and all code/notebooks.
    *   [ ] Ensure code is commented and runnable for the MVP scope.
    *   [ ] Archive project (code, minimal data for reproducibility of MVP, report).
    *   [ ] Project wrap-up & brief presentation of MVP findings.

## Key Milestones (Condensed for 2 Months):

*   **M1 (End of Week 2):** Core data for 1 event acquired and preprocessed. Environment setup complete.
*   **M2 (End of Week 5):** MVP model trained and initially evaluated. Basic XAI method applied and visualizations generated.
*   **M3 (End of Week 8):** Final MVP report, documented code/notebooks, and key visualizations completed and archived.

## Notes for Accelerated Timeline:

*   **Scope Limitation is Critical:** Stick to the defined MVP. Avoid feature creep.
*   **Leverage Existing Resources:** Use pre-trained backbones if possible, adapt existing code snippets, focus on well-understood techniques.
*   **Simplified XAI:** Choose an XAI method that is quick to implement and interpret for the chosen model.
*   **Documentation:** Keep it concise and focused on the MVP.
*   **Vertex AI:** Treat as a stretch goal for a quick test, not a full integration, due to time constraints.
*   **Flexibility:** Be prepared to simplify even further if tasks take longer than expected.

This timeline is very ambitious. Success depends on strict adherence to the limited scope and rapid execution.