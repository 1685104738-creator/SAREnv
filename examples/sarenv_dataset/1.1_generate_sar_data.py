"""Generate the reusable base SAREnv master dataset."""

import json
from pathlib import Path

import numpy as np
from sarenv import (
    CLIMATE_TEMPERATE,
    ENVIRONMENT_TYPE_FLAT,
    DataGenerator,
    get_logger,
)

log = get_logger()

EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = (
    EXAMPLE_DIRECTORY / "sarenv_outputs" / "radiation_area_01"
)


def run_base_dataset_export_example():
    """Export one reusable xlarge base dataset without hazard layers."""
    log.info("--- Starting Base SAREnv Dataset Export Example ---")

    data_gen = DataGenerator()
    initial_planning_point = (-2.66962,51.42351)

    # This is the only example step that accesses Overpass and regenerates the
    # lost-person probability raster.
    data_gen.export_dataset(
        center_point=initial_planning_point,
        output_directory=str(OUTPUT_DIRECTORY),
        environment_climate=CLIMATE_TEMPERATE,
        environment_type=ENVIRONMENT_TYPE_FLAT,
        meter_per_bin=30,
    )

    heatmap_path = OUTPUT_DIRECTORY / "heatmap.npy"
    features_path = OUTPUT_DIRECTORY / "features.geojson"
    metadata_path = OUTPUT_DIRECTORY / "metadata.json"
    expected_paths = (heatmap_path, features_path, metadata_path)
    missing_paths = [path for path in expected_paths if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        message = f"Base dataset export is missing: {missing_text}"
        raise FileNotFoundError(message)

    heatmap = np.load(heatmap_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if tuple(metadata["raster_shape"]) != heatmap.shape:
        raise ValueError(
            "Base metadata raster shape does not match heatmap.npy."
        )

    log.info("--- Base SAREnv dataset exported successfully ---")
    log.info(f"Output directory: {OUTPUT_DIRECTORY}")
    log.info(
        f"Master raster shape: {heatmap.shape}; "
        f"lost-person probability sum: {heatmap.sum():.6f}"
    )
    log.info(
        f"CRS: {metadata['projected_crs']}; "
        f"resolution: {metadata['meter_per_bin']:.1f} m/bin"
    )
    return OUTPUT_DIRECTORY


if __name__ == "__main__":
    run_base_dataset_export_example()
