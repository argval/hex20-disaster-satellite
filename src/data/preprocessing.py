import os
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import pandas as pd
import numpy as np
from google.cloud import storage

# --- Constants --- #
GCS_BUCKET_NAME = 'hex20-disaster-satellite-data'
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
PATCH_SIZE = 256
TRAINING_PATCHES_DIR = 'data/processed/training_patches'
PATCHES_IMAGES_DIR = os.path.join(TRAINING_PATCHES_DIR, 'images')
PATCHES_MASKS_DIR = os.path.join(TRAINING_PATCHES_DIR, 'masks')

# Names of the files as they will be in GCS
SENTINEL_MOSAIC_GCS = 'dixie_fire/sentinel2_mosaic.tif'
FIRMS_POINTS_GCS = 'dixie_fire/firms_points_v2.csv' # Updated to v2

def download_data_from_gcs():
    """Downloads the raw data files from GCS to a local directory."""
    print("--- Starting Data Download from GCS ---")
    try:
        storage_client = storage.Client(project='hex20-disaster-satellite')
        bucket = storage_client.bucket(GCS_BUCKET_NAME)

        # Define source blobs and destination file names
        files_to_download = {
            SENTINEL_MOSAIC_GCS: os.path.join(RAW_DATA_DIR, 'sentinel2_mosaic.tif'),
            FIRMS_POINTS_GCS: os.path.join(RAW_DATA_DIR, 'dixie_fire_firms_points_v2.csv') # Updated local filename
        }

        for gcs_path, local_path in files_to_download.items():
            if os.path.exists(local_path):
                print(f"File already exists, skipping download: {local_path}")
                continue

            print(f"Downloading {gcs_path} to {local_path}...")
            blob = bucket.blob(gcs_path)
            blob.download_to_filename(local_path)
            print(f"Successfully downloaded {local_path}")

        print("--- Data Download Complete ---")
    except Exception as e:
        print(f"An error occurred during GCS download: {e}")
        print("Please ensure your GCS bucket and files exist and you have correct permissions.")

def create_burn_mask():
    """Converts FIRMS points into a rasterized burn mask.

    This mask will have the same dimensions and projection as the
    Sentinel-2 mosaic and will serve as the ground truth label.
    """
    print("\n--- Creating Burn Mask from FIRMS data ---")
    try:
        sentinel_mosaic_path = os.path.join(RAW_DATA_DIR, 'sentinel2_mosaic.tif')
        firms_csv_path = os.path.join(RAW_DATA_DIR, 'dixie_fire_firms_points_v2.csv')
        burn_mask_path = os.path.join(PROCESSED_DATA_DIR, 'burn_mask.tif')

        if not os.path.exists(sentinel_mosaic_path):
            print(f"Error: Sentinel mosaic not found at {sentinel_mosaic_path}")
            print("Please ensure data has been downloaded correctly.")
            return

        if not os.path.exists(firms_csv_path):
            print(f"Error: FIRMS CSV not found at {firms_csv_path}")
            print("Please ensure data has been downloaded correctly.")
            return

        # 1. Read Sentinel-2 mosaic metadata
        with rasterio.open(sentinel_mosaic_path) as src:
            meta = src.meta.copy()
            transform = src.transform
            out_shape = src.shape # (height, width)

        # Ensure meta is suitable for a binary mask
        meta.update(count=1, dtype='uint8', nodata=0) # Binary mask: 0 or 1

        # 2. Read FIRMS data and create a GeoDataFrame
        firms_df = pd.read_csv(firms_csv_path)

        # Verify required columns exist
        if 'longitude' not in firms_df.columns or 'latitude' not in firms_df.columns:
            print("Error: 'longitude' and/or 'latitude' columns not found in FIRMS CSV.")
            print(f"Available columns: {firms_df.columns.tolist()}")
            return

        # Drop rows with missing coordinate data
        firms_df.dropna(subset=['longitude', 'latitude'], inplace=True)

        if firms_df.empty:
            print("Error: No valid fire points with coordinates found after dropping NaNs.")
            return

        # Create a GeoDataFrame directly from lat/lon columns
        # This is much more efficient than parsing strings.
        firms_geo_df = gpd.GeoDataFrame(
            firms_df, 
            geometry=gpd.points_from_xy(firms_df.longitude, firms_df.latitude),
            crs='EPSG:4326' # WGS84 projection, standard for lat/lon
        )

        # 3. Transform FIRMS points to the Sentinel image's CRS
        firms_transformed_gdf = firms_geo_df.to_crs(meta['crs'])

        # 4. Rasterize FIRMS points onto the Sentinel grid
        # We'll create a small buffer around points to make them more visible at Sentinel resolution
        # A buffer of 30m (approx one Sentinel pixel) might be a good start
        # The buffer distance is in the units of the target CRS (usually meters)
        buffered_geometries = firms_transformed_gdf.geometry.buffer(30) # 30 units of the CRS

        print(f"Rasterizing {len(buffered_geometries)} FIRMS features...")
        # Create an empty array for the burn mask
        burn_mask = rasterize(
            shapes=[(geom, 1) for geom in buffered_geometries], # Burn value is 1
            out_shape=out_shape,
            transform=transform,
            fill=0, # Background value
            all_touched=True, # Include pixels touched by the geometry
            dtype='uint8'
        )

        # 5. Save the burn mask as a GeoTIFF
        with rasterio.open(burn_mask_path, 'w', **meta) as dst:
            dst.write(burn_mask.astype(rasterio.uint8), 1)

        print(f"Successfully created burn mask: {burn_mask_path}")
        print("--- Burn Mask Creation Complete ---")

    except Exception as e:
        print(f"An error occurred during burn mask creation: {e}")

def generate_training_patches(patch_size=PATCH_SIZE, min_burn_pixels_ratio=0.01):
    """Crops the large satellite image and burn mask into smaller patches.

    These patches (e.g., 256x256 pixels) are the actual inputs
    that will be fed into the U-Net model during training.
    Only saves patches where the mask contains at least a certain ratio of burned pixels.
    """
    print("\n--- Generating Training Patches ---")
    sentinel_mosaic_path = os.path.join(RAW_DATA_DIR, 'sentinel2_mosaic.tif')
    burn_mask_path = os.path.join(PROCESSED_DATA_DIR, 'burn_mask.tif')

    if not os.path.exists(sentinel_mosaic_path):
        print(f"Error: Sentinel mosaic not found at {sentinel_mosaic_path}")
        return
    if not os.path.exists(burn_mask_path):
        print(f"Error: Burn mask not found at {burn_mask_path}")
        return

    try:
        with rasterio.open(sentinel_mosaic_path) as src_mosaic, rasterio.open(burn_mask_path) as src_mask:
            # Verify compatibility
            if src_mosaic.height != src_mask.height or src_mosaic.width != src_mask.width:
                print("Error: Sentinel mosaic and burn mask have different dimensions.")
                return
            if src_mosaic.crs != src_mask.crs:
                print("Error: Sentinel mosaic and burn mask have different CRS.")
                return
            if src_mosaic.transform != src_mask.transform:
                print("Error: Sentinel mosaic and burn mask have different transforms.")
                return

            mosaic_meta = src_mosaic.meta.copy()
            mask_meta = src_mask.meta.copy()

            width = src_mosaic.width
            height = src_mosaic.height
            saved_patches_count = 0
            skipped_empty_patches_count = 0

            print(f"Source image dimensions: {width}x{height}. Patch size: {patch_size}x{patch_size}")

            for col_off in range(0, width - patch_size + 1, patch_size):
                for row_off in range(0, height - patch_size + 1, patch_size):
                    window = rasterio.windows.Window(col_off, row_off, patch_size, patch_size)
                    
                    # Read mask patch first to check for burn pixels
                    mask_patch = src_mask.read(1, window=window)
                    
                    # Check if the mask patch contains enough burned pixels
                    burned_pixels = np.sum(mask_patch == 1)
                    total_pixels_in_patch = patch_size * patch_size
                    if (burned_pixels / total_pixels_in_patch) < min_burn_pixels_ratio:
                        skipped_empty_patches_count += 1
                        continue # Skip this patch

                    # Read image patch
                    image_patch = src_mosaic.read(window=window)

                    # Update metadata for the patch
                    patch_transform = rasterio.windows.transform(window, src_mosaic.transform)
                    
                    # --- Save Image Patch ---
                    image_patch_meta = mosaic_meta.copy()
                    image_patch_meta.update({
                        'height': patch_size,
                        'width': patch_size,
                        'transform': patch_transform
                    })
                    image_patch_filename = os.path.join(PATCHES_IMAGES_DIR, f'patch_{col_off}_{row_off}.tif')
                    with rasterio.open(image_patch_filename, 'w', **image_patch_meta) as dst:
                        dst.write(image_patch)

                    # --- Save Mask Patch ---
                    mask_patch_meta = mask_meta.copy()
                    mask_patch_meta.update({
                        'height': patch_size,
                        'width': patch_size,
                        'transform': patch_transform
                    })
                    mask_patch_filename = os.path.join(PATCHES_MASKS_DIR, f'patch_{col_off}_{row_off}.tif')
                    with rasterio.open(mask_patch_filename, 'w', **mask_patch_meta) as dst:
                        dst.write(mask_patch, 1)

                    saved_patches_count += 1

                    # Save patches
                    patch_filename_base = f"patch_{row_off}_{col_off}.tif"
                    img_patch_path = os.path.join(PATCHES_IMAGES_DIR, patch_filename_base)
                    msk_patch_path = os.path.join(PATCHES_MASKS_DIR, patch_filename_base)

                    with rasterio.open(img_patch_path, 'w', **mosaic_patch_meta) as dst_img:
                        dst_img.write(image_patch)
                    
                    with rasterio.open(msk_patch_path, 'w', **mask_patch_meta) as dst_msk:
                        dst_msk.write(mask_patch, 1)
                    
                    saved_patches_count += 1
                    if saved_patches_count % 100 == 0:
                        print(f"Saved {saved_patches_count} patches... (skipped {skipped_empty_patches_count} empty)")

            print(f"Successfully generated {saved_patches_count} training patches.")
            print(f"Skipped {skipped_empty_patches_count} patches due to insufficient burn pixels.")
            print(f"Image patches saved to: {PATCHES_IMAGES_DIR}")
            print(f"Mask patches saved to: {PATCHES_MASKS_DIR}")

    except Exception as e:
        print(f"An error occurred during training patch generation: {e}")
    
    print("--- Training Patch Generation Complete ---")

def main():
    """Main preprocessing pipeline."""
    # Create local directories if they don't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(PATCHES_IMAGES_DIR, exist_ok=True)
    os.makedirs(PATCHES_MASKS_DIR, exist_ok=True)

    # 1. Download data from GCS
    download_data_from_gcs()

    # 2. Create the burn mask from FIRMS points
    create_burn_mask()

    # 3. Generate training patches
    generate_training_patches()

if __name__ == '__main__':
    main()
