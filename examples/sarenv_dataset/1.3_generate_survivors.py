from pathlib import Path

from sarenv import (
    DatasetLoader,
    LostPersonLocationGenerator,
    get_logger,
    load_lost_person_locations,
    save_lost_person_locations,
)

log = get_logger()

if __name__ == "__main__":
    log.info("--- Starting lost_person Location Generation Example for Custom Area ---")

    example_directory = Path(__file__).resolve().parent
    dataset_dir = example_directory / "sarenv_outputs" / "radiation_area_01"
    num_locations = 100

    try:
        # The saved metadata is authoritative for the physical dataset size.
        loader = DatasetLoader(dataset_directory=str(dataset_dir))
        dataset_item = loader.load_environment()

        if not dataset_item:
            raise RuntimeError("Could not load the saved SAR dataset.")
        log.info(f"Loaded saved environment size: '{dataset_item.size}'")

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
            output_path = save_lost_person_locations(
                locations,
                dataset_item,
                dataset_dir,
            )
            restored = load_lost_person_locations(output_path)
            if len(restored.points) != len(locations):
                raise ValueError("Lost-person save/load count mismatch.")
            log.info(f"Saved and reloaded locations: {output_path}")

    except FileNotFoundError:
        log.error(
            f"Error: The dataset directory '{dataset_dir}' or its master files were not found."
        )
        log.error(
            "Please run the custom DataGenerator example first."
        )
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)
