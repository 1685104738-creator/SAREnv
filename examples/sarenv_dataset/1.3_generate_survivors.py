from sarenv import (
    DatasetLoader,
    LostPersonLocationGenerator,
    get_logger,
)

log = get_logger()

if __name__ == "__main__":
    log.info("--- Starting lost_person Location Generation Example for Custom Area ---")

    dataset_dir = "sarenv_outputs/test_area_01"
    size_to_load = "xlarge"
    num_locations = 100

    try:
        # 1. Load the custom dataset for a specific size
        log.info(f"Loading data for size: '{size_to_load}'")
        loader = DatasetLoader(dataset_directory=dataset_dir)
        dataset_item = loader.load_environment(size_to_load)

        if not dataset_item:
            log.error(f"Could not load the dataset for size '{size_to_load}'.")

        # 2. Initialize the lost_person location generator with the loaded data
        log.info("Initializing the lost_person LocationGenerator.")
        lost_person_generator = LostPersonLocationGenerator(dataset_item)

        # 3. Generate lost_person locations
        log.info(f"Generating {num_locations} lost_person locations...")
        locations = lost_person_generator.generate_locations(num_locations, 0)  # 0% random samples

        if not locations:
            log.error("No lost_person locations were generated.")
        else:
            log.info(f"Successfully generated {len(locations)} lost_person locations.")

    except FileNotFoundError:
        log.error(
            f"Error: The dataset directory '{dataset_dir}' or its master files were not found."
        )
        log.error(
            "Please run the custom DataGenerator example first."
        )
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)