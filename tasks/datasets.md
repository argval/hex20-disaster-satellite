# Datasets for Wildfire Detection and Monitoring

This document outlines the primary and supplementary datasets planned for use in this project.

## 1. Primary Satellite Data

*   **Sentinel-2 (Optical Imagery):**
    *   **Source:** Copernicus Programme (ESA), accessed via Google Earth Engine (`COPERNICUS/S2_SR_HARMONIZED` for Surface Reflectance).
    *   **Spatial Resolution:** 10m (Visible, NIR), 20m (Red Edge, SWIR), 60m (Atmospheric bands).
    *   **Temporal Resolution:** ~5 days (with Sentinel-2A and -2B).
    *   **Key Bands for Wildfire:**
        *   Visible (B2, B3, B4): For true-color composites and visual assessment.
        *   Near Infrared (B8, B8A): Vegetation health, biomass, crucial for NDVI and NBR.
        *   Short-Wave Infrared (B11, B12): Sensitive to moisture content, soil, and burned areas; crucial for NBR and other burn indices.
    *   **Derived Products:**
        *   Normalized Difference Vegetation Index (NDVI): (NIR - Red) / (NIR + Red)
        *   Normalized Burn Ratio (NBR): (NIR - SWIR2) / (NIR + SWIR2)
        *   Difference NBR (dNBR): NBR_prefire - NBR_postfire (for burn severity).
    *   **Use Cases:** Burned area mapping, burn severity assessment, vegetation monitoring pre/post-fire.

*   **FIRMS - Fire Information for Resource Management System (Active Fires/Thermal Anomalies):**
    *   **Source:** NASA, accessed via Google Earth Engine (`FIRMS` collection) or NASA FIRMS API/downloads.
    *   **Sensors:**
        *   MODIS (Moderate Resolution Imaging Spectroradiometer): On Terra and Aqua satellites. ~1km spatial resolution for fire detection.
        *   VIIRS (Visible Infrared Imaging Radiometer Suite): On Suomi NPP and NOAA-20 satellites. ~375m spatial resolution for fire detection.
    *   **Temporal Resolution:** Multiple observations per day per sensor.
    *   **Key Attributes:** Fire pixel location (latitude, longitude), brightness temperature, confidence level, acquisition date/time, satellite.
    *   **Use Cases:** Active fire detection, hotspot mapping, near real-time fire monitoring, input for fire spread models, validation of burned area products.

## 2. Meteorological Data

*   **ERA5 (Weather Reanalysis Data):**
    *   **Source:** ECMWF (European Centre for Medium-Range Weather Forecasts), accessed via Google Earth Engine (`ECMWF/ERA5/DAILY` or `ECMWF/ERA5_LAND/HOURLY`).
    *   **Spatial Resolution:** ~0.25 degrees (~25-30 km) for ERA5, ~0.1 degrees (~9km) for ERA5-Land.
    *   **Temporal Resolution:** Daily or Hourly.
    *   **Key Variables for Wildfire Context:**
        *   2m Temperature (mean, min, max)
        *   2m Dewpoint Temperature / Relative Humidity
        *   Total Precipitation
        *   Surface Pressure
        *   10m Wind Speed and Direction (U and V components)
        *   Volumetric Soil Water
        *   Potentially derived fire weather indices (e.g., FWI components if calculated or data available).
    *   **Use Cases:** Understanding weather conditions conducive to fire ignition and spread, contextual data for ML models, calculating fire danger indices.

## 3. Ancillary/Validation Datasets (To be explored)

*   **Global Forest Change (Hansen et al.):**
    *   **Source:** University of Maryland, accessed via Google Earth Engine (`UMD/hansen/global_forest_change_20XX`).
    *   **Content:** Tree cover, forest loss/gain.
    *   **Use Cases:** Land cover information, assessing fire impact on forests.

*   **Copernicus Global Land Cover:**
    *   **Source:** Copernicus Land Monitoring Service.
    *   **Content:** Discrete and fractional land cover classifications.
    *   **Use Cases:** Land cover context for fire analysis, stratification of results.

*   **Digital Elevation Models (DEMs):**
    *   **Source:** SRTM (Shuttle Radar Topography Mission - `USGS/SRTMGL1_003`), Copernicus DEM (`COPERNICUS/DEM/GLO30`).
    *   **Content:** Elevation, slope, aspect.
    *   **Use Cases:** Topographic context, influence on fire behavior.

*   **Historical Fire Perimeters/Databases:**
    *   **Source:** National/regional agencies (e.g., MTBS in the US, EFFIS in Europe, state fire agencies).
    *   **Content:** Polygons of historical fire events.
    *   **Use Cases:** Ground truth for training and validating burned area mapping models. This is often the most critical and challenging dataset to acquire consistently across regions.

*   **CAL FIRE Fire Perimeters:**
    *   **Source:** California Department of Forestry and Fire Protection.
    *   **Use Cases:** Specific ground truth for California wildfires.

*   **MTBS (Monitoring Trends in Burn Severity):**
    *   **Source:** USGS/USFS.
    *   **Content:** Burned area boundaries and severity classifications for fires >1000 acres (US).
    *   **Use Cases:** Validation data for burned area and severity in the US.

## 4. Additional Datasets for Investigation (from XAI Research Documents)

These datasets were highlighted in the provided research documents and will be investigated for their potential integration and utility.

*   **Global Forest Burn Severity (GFBS) Dataset:**
    *   **Source:** Copernicus (via ESSD article link provided).
    *   **Description:** 30m resolution global forest burn severity (2003-2016).
    *   **Format:** GeoTIFF.
    *   **Potential Use Cases:** Training/validating burn severity models, providing historical context on burn severity patterns.

*   **Google FireSat:**
    *   **Source:** Google Research (via research site link).
    *   **Description:** High-resolution multispectral satellite imagery with AI analysis for near real-time insights.
    *   **Access:** Likely through Google Earth Engine; specific terms and availability for research to be confirmed.
    *   **Potential Use Cases:** Near real-time fire detection, high-resolution fire progression analysis.

*   **NOAA Hazard Mapping System (HMS) Fire and Smoke Product:**
    *   **Source:** NOAA OSPO.
    *   **Description:** Real-time satellite analysis of smoke, fire, and dust.
    *   **Format:** Shapefile, KML, WMS.
    *   **Potential Use Cases:** Contextual information on smoke plumes, validation of active fire detections, potential for smoke segmentation tasks.

## 5. Publicly Available Wildfire Datasets for ML (Examples from provided list)

*   **Zenodo, Kaggle, GitHub, SpaceML, DrivenData:** These platforms host various curated datasets that may include satellite imagery (Landsat, Sentinel, MODIS) paired with fire labels. These will be investigated for:
    *   Pre-training models.
    *   Benchmarking.
    *   Understanding data structures and labeling conventions.
    *   Examples:
        *   "Wildfire Dataset from Multi-Sensor Satellite Imagery" (Zenodo)
        *   "California Wildfires" (Kaggle)
        *   Various datasets on SpaceML.ai related to fire.

**Data Management Note:** All downloaded or processed data will be organized systematically. Consideration will be given to using cloud storage for larger datasets and tools like DVC for versioning.