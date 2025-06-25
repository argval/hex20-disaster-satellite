from src.data.preprocessing import download_data_from_gcs, create_burn_mask, generate_training_patches

if __name__ == '__main__':
    download_data_from_gcs()
    create_burn_mask()
    generate_training_patches()
