# Research and References: Satellite-Based Wildfire Detection and Monitoring

This document outlines key research areas, relevant literature (to be expanded), and important technologies for the project.

## 1. Key Research Areas and Questions

*   **Data Fusion Techniques:**
    *   How can optical (Sentinel-2), thermal (FIRMS), and meteorological (ERA5) data be optimally fused to improve wildfire detection accuracy and timeliness?
    *   What are the best methods for handling differences in spatial and temporal resolutions?
*   **Machine Learning for Burned Area Mapping:**
    *   Which deep learning architectures (e.g., U-Net, SegNet, DeepLabv3+, Transformers) are most effective for segmenting burned areas from Sentinel-2 imagery?
    *   How can temporal information (pre- and post-fire imagery) be best incorporated?
    *   Investigating the utility of spectral indices (NBR, NDVI, etc.) as input features vs. raw band data.
*   **Machine Learning for Active Fire Detection:**
    *   Can ML models improve upon traditional threshold-based methods for active fire detection using FIRMS and Sentinel-2 data?
    *   How to minimize false positives in active fire detection?
*   **Burn Severity Assessment:**
    *   Relationship between dNBR (and other indices) and field-observed burn severity.
    *   Can ML models directly estimate burn severity levels?
*   **Time-Series Analysis for Fire Monitoring and Prediction:**
    *   Using time-series of satellite data to monitor fire progression.
    *   Exploring the potential for short-term fire spread prediction using ML and integrated datasets.
*   **Generalization and Transfer Learning:**
    *   How well do models trained on one region/fire regime generalize to others?
    *   Can pre-trained models (e.g., on general remote sensing datasets or ImageNet) improve performance and reduce training data requirements?
*   **Cloud Computing and Big Data:**
    *   Leveraging Google Earth Engine for efficient large-scale data access and preprocessing.
    *   Strategies for managing and processing large volumes of satellite imagery.
*   **Resource-Efficient Model Architectures:**
    *   How to design models that balance accuracy with computational cost (CPU, memory, latency) for potential deployment in resource-constrained environments (e.g., edge devices)?
    *   What are the trade-offs of techniques like pruning, quantization, and knowledge distillation for wildfire detection models?
*   **Efficient and Trustworthy Explainable AI (XAI):**
    *   What are the most suitable and computationally feasible XAI techniques (e.g., SHAP/LIME approximations, rule extraction, saliency maps) for our wildfire models?
    *   How can XAI outputs be effectively visualized and communicated to end-users to build trust and aid decision-making?
    *   How to evaluate the faithfulness and utility of explanations in the context of wildfire monitoring?

## 2. Core Technologies and Tools

*   **Programming Language:** Python
*   **Geospatial Libraries:**
    *   `rasterio`: Reading, writing, and manipulating raster data.
    *   `geopandas`: Working with vector data (e.g., fire perimeters, FIRMS points).
    *   `shapely`: Geometric operations.
    *   `pyproj`: Coordinate reference system transformations.
    *   `xarray`: Working with N-dimensional labeled arrays, suitable for satellite data cubes.
*   **Data Handling & Numerics:**
    *   `numpy`: Fundamental package for numerical computation.
    *   `pandas`: Data analysis and manipulation, especially for tabular data.
*   **Machine Learning Frameworks:**
    *   `pytorch` (preferred): Deep learning framework.
    *   `tensorflow`/`keras`: Alternative deep learning framework.
    *   `scikit-learn`: General machine learning algorithms, evaluation metrics.
    *   `segmentation-models-pytorch`: Pre-trained segmentation models and building blocks.
*   **Earth Observation Platforms:**
    *   `earthengine-api` (Python client for Google Earth Engine): Accessing and processing petabytes of satellite imagery and geospatial datasets.
*   **Visualization:**
    *   `matplotlib`: Standard Python plotting library.
    *   `seaborn`: Statistical data visualization.
    *   `folium`: Interactive maps using Leaflet.js.
    *   `geemap`: Python package for interactive mapping with Google Earth Engine, built on `ipyleaflet` and `folium`.
*   **Development Environment:**
    *   Jupyter Notebooks / JupyterLab: Interactive development and analysis.
    *   VS Code (or other IDE) with Python support.
*   **Version Control:** Git, GitHub/GitLab.
*   **Data Versioning (Optional but Recommended):** DVC (Data Version Control).
*   **Dependency Management:** `pip`, `conda`, `pyproject.toml` (Poetry or Hatch).
*   **Cloud AI Platform:** `google-cloud-aiplatform` (Vertex AI SDK for Python).
*   **Explainable AI Services:** Google Cloud Vertex AI Explainable AI.

## 3. Literature Review (To be populated - placeholders)

*   **Seminal Papers on Remote Sensing of Wildfires:**
    *   [Placeholder for key review articles on satellite fire monitoring]
    *   [Placeholder for foundational papers on NBR, dNBR]
*   **Deep Learning for Burned Area Mapping:**
    *   [Placeholder for U-Net applications in remote sensing/burned area]
    *   [Placeholder for recent papers using CNNs/Transformers for segmentation]
    *   Example: Knopp et al. (2020) - "Deep Learning for Wildfire Detection and Segmentation from Satellite Imagery"
*   **Active Fire Detection with ML:**
    *   [Placeholder for papers on ML for active fire points/pixels]
*   **Data Fusion in Remote Sensing:**
    *   [Placeholder for review articles on multi-sensor data fusion]
*   **Specific Datasets and Benchmarks:**
    *   [Placeholder for papers introducing relevant public wildfire datasets]
*   **Fire Weather Indices and their Application:**
    *   [Placeholder for literature on FWI, McArthur Forest Fire Danger Index, etc.]

## 4. Key Insights from Internal Research Documents (User-Provided PDFs)

*   **Multi-Source Explainable Wildfire Detection for Resource-Constrained Environments:**
    *   **Focus:** Developing lightweight, explainable wildfire detection systems integrating multiple data sources, optimized for resource-constrained settings.
    *   **Key Ideas:** Comparative analysis of detection methods, lightweight model architectures (pruning, quantization, distillation), multi-source data fusion, efficient XAI techniques (approximate SHAP/LIME, rule extraction), benchmarking against complex models.
    *   **Relevance:** Directly informs our XAI goals, lightweight model development, and evaluation strategy.

*   **Technical Requirements and Dependencies for Satellite Imaging Disaster Monitoring Project:**
    *   **Focus:** Outlines essential cloud platforms (GEE, Colab), Python libraries (core, geospatial, EE API), M1 Mac considerations, and development tools.
    *   **Relevance:** Confirms and expands our existing tech stack, provides setup guidance, and highlights best practices for data/memory management.

*   **Recommended Datasets for Satellite Imaging Disaster Monitoring:**
    *   **Focus:** Lists key datasets for forest fires (FIRMS, GFBS, Google FireSat, NOAA HMS), floods, and hurricanes.
    *   **Relevance:** Introduces new potential datasets (GFBS, FireSat, HMS) for our wildfire project and provides context for future expansion to other disaster types.

*   **Edge Computing for Wildfire Detection: Conceptual Framework and Research Directions:**
    *   **Focus:** Proposes a conceptual framework for applying edge computing to wildfire detection, including hardware, software stack, edge-cloud hybrid architecture, data processing optimizations, and research opportunities.
    *   **Key Ideas:** Tiered processing (edge-fog-cloud), model optimization for edge (quantization, pruning, splitting), adaptive resource allocation, collaborative edge detection, ultra-lightweight XAI for edge.
    *   **Relevance:** Provides a roadmap for an optional future phase focused on edge deployment, aligning with lightweight model goals.

## 5. Useful Online Resources and Communities

*   **Google Earth Engine:**
    *   [Earth Engine Documentation](https://developers.google.com/earth-engine)
    *   [Earth Engine Community Forum](https://groups.google.com/g/google-earth-engine-developers)
*   **NASA FIRMS:**
    *   [FIRMS Website](https://firms.modaps.eosdis.nasa.gov/)
*   **Copernicus Programme:**
    *   [Sentinel Online](https://sentinels.copernicus.eu/)
*   **ECMWF ERA5:**
    *   [ERA5 Data Documentation](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
*   **PyTorch & TensorFlow Documentation:**
    *   [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
    *   [TensorFlow Docs](https://www.tensorflow.org/api_docs)
*   **Relevant Conferences and Journals:**
    *   IEEE Transactions on Geoscience and Remote Sensing (TGRS)
    *   Remote Sensing of Environment (RSE)
    *   ISPRS Journal of Photogrammetry and Remote Sensing
    *   CVPR, ICCV, NeurIPS, ICML (for general ML/CV advancements)
    *   IGARSS (International Geoscience and Remote Sensing Symposium)

This document will be updated continuously as the project progresses and new information is gathered.