# Week 1 Tasks: MVP Project Kickoff (Early June 2025)

This document outlines the key objectives for Week 1 of the accelerated 2-month Wildfire Detection MVP project.

## Objectives:

1.  **[X] Solidify MVP Scope:**
    *   [X] Define one specific wildfire event/region for analysis.
        *   **Decision:** California selected as the region.
        *   **Final Selection:** The Dixie Fire (2021) is chosen as the primary focus for the MVP.

2.  **[X] Confirm Primary Datasets:**
    *   [X] Sentinel-2 L2A imagery.
    *   [X] FIRMS active fire data.
    *   **Note:** These are confirmed as the sole datasets for the MVP.

3.  **[X] Set Up Project Environment:**
    *   [X] Git repository established.
    *   [X] Python virtual environment managed (e.g., via `uv`).
    *   [X] Google Earth Engine authentication successful.
    *   [X] NASA Earthdata authentication successful.
    *   [X] Verify installation of core libraries:
        *   `rasterio` (Verified via pyproject.toml)
        *   `geopandas` (Verified via pyproject.toml)
        *   `pytorch` (Selected due to segmentation-models-pytorch, verified via pyproject.toml)
        *   `scikit-learn` (Verified via pyproject.toml)
        *   `segmentation-models-pytorch` (Verified via pyproject.toml)

4.  **[X] Basic Literature Check for Model Architecture:**
    *   [X] Identify a suitable, pre-existing, well-documented model architecture for burned area mapping.
        *   **Decision:** U-Net architecture, likely from the `segmentation-models-pytorch` library.
    *   **Goal:** Select a standard architecture to adapt, not to design from scratch. (Achieved)

## End of Week 1 Goal:
- A specific wildfire event in California is chosen.
- All necessary development environment components are confirmed to be in place.
- A candidate ML model architecture is identified.
