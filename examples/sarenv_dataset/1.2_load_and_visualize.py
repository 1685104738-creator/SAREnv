from sarenv import (
    DatasetLoader,
    get_logger,
    visualize_heatmap,
    visualize_features,
)

log = get_logger()


def run_custom_loading_example():
    """
    Load and visualize the custom dataset generated in case 1.
    This checks whether the new area can be used by SAREnv's standard loading
    and visualization pipeline.
    """
    log.info("--- Starting Custom Dataset Loading and Visualization Example ---")

    # This must match the output_dir used in case 1.
    dataset_dir = "sarenv_outputs/test_area_01"

    # Start with xlarge because case 1 generated the master xlarge dataset.
    size_to_load = "xlarge"

    try:
        loader = DatasetLoader(dataset_directory=dataset_dir)

        log.info(f"Loading custom dataset from: {dataset_dir}")
        log.info(f"Loading data for size: '{size_to_load}'")

        item = loader.load_environment(size_to_load)

        if item:
            log.info("--- Dataset loaded successfully ---")
            log.info(f"Center point: {item.center_point}")
            log.info(f"Radius: {item.radius_km} km")
            log.info(f"Heatmap shape: {item.heatmap.shape}")
            log.info(f"Heatmap probability sum: {item.heatmap.sum():.6f}")
            log.info(f"Number of clipped features: {len(item.features)}")
            log.info(f"Environment climate: {item.environment_climate}")
            log.info(f"Environment type: {item.environment_type}")

            # Avoid basemap first. This keeps the test simpler and avoids extra online map downloads.
            visualize_heatmap(item, plot_basemap=False, plot_inset=True)
            visualize_features(item, plot_basemap=False, plot_inset=True, num_lost_persons=300)

        else:
            log.error(f"Could not load the specified size: '{size_to_load}'")

    except FileNotFoundError:
        log.error(
            f"Error: The dataset directory '{dataset_dir}' or its master files were not found."
        )
        log.error(
            "Please run case 1 first to generate features.geojson and heatmap.npy."
        )

    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    run_custom_loading_example()