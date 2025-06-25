import ee
import os

# Initialize the Earth Engine API
# This assumes you have authenticated via `earthengine authenticate`
try:
    ee.Initialize(project='hex20-disaster-satellite')
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing Earth Engine: {e}")
    print("Please ensure you have authenticated with a project using 'earthengine authenticate'.")

# --- Constants for the Dixie Fire MVP --- #

# Date range for the analysis. Focusing on the main period of the fire.
DIXIE_FIRE_START_DATE = '2021-07-13'
DIXIE_FIRE_END_DATE = '2021-10-01' # Giving a buffer past the main activity

# Counties affected by the Dixie Fire. We'll use this to create a bounding box.
# Source: Wikipedia
AFFECTED_COUNTIES = [
    'Butte',
    'Plumas',
    'Lassen',
    'Shasta',
    'Tehama'
]

# Google Cloud Storage bucket for exporting data.
# IMPORTANT: You must create this bucket in your Google Cloud project.
GCS_BUCKET = 'hex20-disaster-satellite-data' # Example bucket name


def get_dixie_fire_roi():
    """Creates a Region of Interest (ROI) for the Dixie Fire.

    For the MVP, this will be a bounding box encompassing the affected counties.
    A more advanced approach would use the final fire perimeter.

    Returns:
        ee.Geometry: The region of interest for the Dixie Fire.
    """
    # Load US counties feature collection.
    counties = ee.FeatureCollection('TIGER/2018/Counties')

    # Filter to California counties.
    ca_counties = counties.filter(ee.Filter.eq('STATEFP', '06'))

    # Filter to the specific counties affected by the Dixie Fire.
    dixie_counties = ca_counties.filter(ee.Filter.inList('NAME', AFFECTED_COUNTIES))

    # Get the union of the county geometries to form a single ROI.
    # The .dissolve() method is efficient for this.
    roi = dixie_counties.union().geometry()

    print("Successfully created ROI for the Dixie Fire.")
    return roi

def mask_s2_clouds(image):
    """Masks clouds in a Sentinel-2 image using the QA60 band.

    Args:
        image (ee.Image): A Sentinel-2 image.

    Returns:
        ee.Image: Cloud-masked Sentinel-2 image.
    """
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0) \
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    return image.updateMask(mask).divide(10000)

def get_sentinel2_collection(roi, start_date, end_date):
    """Acquires and filters a Sentinel-2 image collection.

    Args:
        roi (ee.Geometry): The region of interest.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        ee.ImageCollection: The filtered and cloud-masked Sentinel-2 collection.
    """
    s2_collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        # Pre-filter to get less cloudy scenes.
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mask_s2_clouds)
    )

    # Print the number of images found.
    count = s2_collection.size().getInfo()
    print(f"Found {count} Sentinel-2 images.")

    return s2_collection

def get_firms_data(roi, start_date, end_date):
    """Acquires and filters FIRMS active fire data, adding lat/lon properties.

    Args:
        roi (ee.Geometry): The region of interest.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        ee.FeatureCollection: The filtered FIRMS data with added lat/lon.
    """
    # FIRMS is an ImageCollection, where each image represents a day's observations.
    firms_image_collection = (
        ee.ImageCollection('FIRMS')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .select(['confidence', 'T21']) # Select relevant bands
    )

    # Function to process each daily FIRMS image
    def extract_fire_points_from_image(image):
        # A common threshold for FIRMS confidence is 'nominal' or 'high'.
        # The 'confidence' band is often categorical (e.g., 7, 8, 9 for low, nominal, high)
        # or numerical (0-100). For MODIS/VIIRS, 'nominal' (value 8) and 'high' (value 9) are often used.
        # Let's assume values >= 8 represent nominal or high confidence fires.
        # This might need adjustment based on specific FIRMS product version details.
        fire_mask = image.select('confidence').gte(8) # pixels with confidence >= 8 (nominal or high)
        
        # Mask the image to keep only fire pixels
        active_fire_pixels = image.updateMask(fire_mask)

        # Convert fire pixels to points. Scale should be appropriate for FIRMS (approx 1km)
        # We need to reproject to a known CRS like EPSG:4326 if we want lat/lon easily.
        # However, reduceToVectors works on the image's native projection first.
        fire_points = active_fire_pixels.reduceToVectors(
            reducer=ee.Reducer.firstNonNull(), # Keep properties of the first band found
            geometry=roi, # Clip to our ROI
            scale=1000, # FIRMS nominal resolution is 1km
            geometryType='centroid',
            eightConnected=False,
            labelProperty='confidence_value', # The pixel value will be in this property
            # crs='EPSG:4326' # Optional: reproject during vectorization
        )

        # Function to add lat/lon to each point feature
        def add_lat_lon_to_point(feature):
            geom = feature.geometry() # This should be a point
            coords = ee.List(geom.coordinates()) # [lon, lat]
            # Ensure it's a valid point with two coordinates
            is_valid_point = coords.size().eq(2)
            lon = ee.Algorithms.If(is_valid_point, coords.get(0), None)
            lat = ee.Algorithms.If(is_valid_point, coords.get(1), None)
            return feature.set({'longitude': lon, 'latitude': lat, 'original_confidence': feature.get('confidence_value')})

        # Filter out null geometries just in case, then add lat/lon
        return fire_points.filter(ee.Filter.geometry(roi)).map(add_lat_lon_to_point)

    # Map over the image collection and flatten the result into a single FeatureCollection
    all_fire_points = firms_image_collection.map(extract_fire_points_from_image).flatten()
    
    # Filter out any points where lat/lon extraction might have failed (though less likely now)
    all_fire_points = all_fire_points.filter(ee.Filter.And(
        ee.Filter.neq('longitude', None),
        ee.Filter.neq('latitude', None)
    ))

    count = all_fire_points.size().getInfo()
    print(f"Found {count} FIRMS active fire points after processing daily images.")

    return all_fire_points

def export_sentinel_to_gcs(collection, roi, bucket):
    """Creates a mosaic and exports it to Google Cloud Storage.

    Args:
        collection (ee.ImageCollection): Sentinel-2 collection.
        roi (ee.Geometry): The region of interest for clipping.
        bucket (str): The GCS bucket name.
    """
    # Create a median composite.
    mosaic = collection.median()

    # Define the export task.
    task = ee.batch.Export.image.toCloudStorage(
        image=mosaic.clip(roi).select(['B4', 'B3', 'B2', 'B8', 'B12']), # RGB, NIR, SWIR
        description='DixieFire_Sentinel2_Mosaic',
        bucket=bucket,
        fileNamePrefix='dixie_fire/sentinel2_mosaic',
        scale=30, # 30m resolution for simplicity, can be 10 or 20
        region=roi,
        maxPixels=1e13
    )

    task.start()
    print(f"Started GCS export task for Sentinel-2 mosaic: {task.id}")

def export_firms_to_gcs(collection, bucket):
    """Exports a feature collection with specific selectors to Google Cloud Storage.

    Args:
        collection (ee.FeatureCollection): FIRMS collection (should have lat/lon).
        bucket (str): The GCS bucket name.
    """
    # Define the properties we want to export
    # Common FIRMS properties: 'T21' (brightness K), 'confidence', 'frp', 'acq_date', 'acq_time'
    # Ensure 'latitude' and 'longitude' are from our add_lat_lon map function.
    selectors = ['system:index', 'latitude', 'longitude', 'T21', 'confidence', 'frp', 'acq_date', 'acq_time']

    task = ee.batch.Export.table.toCloudStorage(
        collection=collection,
        description='DixieFire_FIRMS_Points_v2',
        bucket=bucket,
        fileNamePrefix='dixie_fire/firms_points_v2', # New filename
        fileFormat='CSV',
        selectors=selectors
    )

    task.start()
    print(f"Started GCS export task for FIRMS data (v2 with selectors): {task.id}")

def acquire_data_for_dixie_fire():
    """Main function to acquire Sentinel-2 and FIRMS data for the Dixie Fire.

    This function will orchestrate the following steps:
    1. Define the Region of Interest (ROI).
    2. Filter Sentinel-2 imagery for the ROI and date range.
    3. Filter FIRMS data for the ROI and date range.
    4. Prepare and initiate export tasks to Google Cloud Storage.
    """
    print("--- Starting Data Acquisition for Dixie Fire MVP ---")

    # 1. Get the Region of Interest
    roi = get_dixie_fire_roi()

    # 2. Get Sentinel-2 Data
    print("\nStep 2: Acquiring Sentinel-2 Imagery...")
    sentinel_collection = get_sentinel2_collection(roi, DIXIE_FIRE_START_DATE, DIXIE_FIRE_END_DATE)
    # The export_sentinel_to_gcs function will handle mosaicking from the collection.

    # 3. Get the FIRMS Active Fire Data
    print("\nStep 3: Acquiring FIRMS Active Fire Data...")
    firms_collection = get_firms_data(roi, '2021-07-01', '2021-08-31')

    # 4. Export Data to Google Cloud Storage
    print("\nStep 4: Initiating Export Tasks to GCS...")
    export_sentinel_to_gcs(sentinel_collection, roi, GCS_BUCKET) # Use GCS_BUCKET
    export_firms_to_gcs(firms_collection, GCS_BUCKET) # Use GCS_BUCKET

    print("\n--- Data Acquisition Script Complete ---")
    print("Please monitor the task status in your Earth Engine account:")
    print("https://code.earthengine.google.com/tasks")

if __name__ == '__main__':
    acquire_data_for_dixie_fire()
