import os

import geopandas as gpd
import numpy as np
from pathlib import Path
import shapely
from sarenv import (
    CLIMATE_DRY,
    CLIMATE_TEMPERATE,
    ENVIRONMENT_TYPE_FLAT,
    ENVIRONMENT_TYPE_MOUNTAINOUS,
    DataGenerator,
    get_logger,
)

log = get_logger()

def run_export_example():
    """
    An example function demonstrating how to use the DataGenerator
    to export features and heatmaps for all quantiles.
    """
    log.info("--- Starting DataGenerator Export Example ---")

    # 1. Initialize the generator.
    data_gen = DataGenerator()

    # 2. Define a center point and an output directory for the dataset.
    initial_planning_point =  -2.66962,51.42351
    output_dir = "sarenv_outputs/test_area_01"

    # 3. Run the main export function.
    data_gen.export_dataset(
        center_point=initial_planning_point,
        output_directory=output_dir,
        environment_climate=CLIMATE_TEMPERATE,
        environment_type=ENVIRONMENT_TYPE_FLAT,
        meter_per_bin=30,
    )

    log.info("--- Verifying exported files ---")
    try:
        # Check the files for the 'median' quantile
        master_heatmap_path = os.path.join(output_dir, "heatmap.npy")
        master_features_path = os.path.join(output_dir, "features.geojson")

        if os.path.exists(master_heatmap_path):
            heatmap_matrix = np.load(master_heatmap_path)
            log.info(f"Loaded heatmap 'heatmap.npy'. Shape: {heatmap_matrix.shape}")
            # You could now use this matrix for analysis or as input to a model.
        else:
            log.error(f"Verification failed: {master_heatmap_path} not found.")

        if os.path.exists(master_features_path):
            features_gdf = gpd.read_file(master_features_path)
            log.info(
                f"Loaded features 'features.geojson'. Found {len(features_gdf)} features."
            )
            log.info("Sample of loaded features:")
            print(features_gdf.head())
        else:
            log.error(f"Verification failed: {master_features_path} not found.")

    except Exception as e:
        log.error(f"An error occurred during verification: {e}", exc_info=True)


if __name__ == "__main__":
    run_export_example()
