"""Generate or replace a radiation layer for an existing base dataset."""

import json
from pathlib import Path

import numpy as np
from sarenv import (
    RadiationConfig,
    generate_radiation_layer,
    get_logger,
)


log = get_logger()
EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
BASE_DATASET_DIRECTORY = (
    EXAMPLE_DIRECTORY / "sarenv_outputs" / "radiation_area_01"
)


def run_radiation_layer_example():
    """Generate one offline synthetic radiation scenario on the base grid."""
    log.info("--- Starting Offline Radiation Layer Example ---")

    radiation_config = RadiationConfig(
        source_type="distributed",
        placement_type="near",
        intensity_type="medium",
        shape_type="stretched",
        seed=42,
    )
    result = generate_radiation_layer(
        BASE_DATASET_DIRECTORY,
        radiation_config,
    )

    heatmap = np.load(
        BASE_DATASET_DIRECTORY / "heatmap.npy",
        allow_pickle=False,
    )
    radiation = np.load(
        BASE_DATASET_DIRECTORY / "radiation.npy",
        allow_pickle=False,
    )
    metadata = json.loads(
        (BASE_DATASET_DIRECTORY / "radiation_meta.json").read_text(
            encoding="utf-8"
        )
    )
    if radiation.shape != heatmap.shape:
        raise ValueError(
            "Offline radiation raster does not match the base heatmap shape."
        )

    log.info("--- Radiation layer generated without rebuilding SAREnv ---")
    log.info(f"Base dataset: {BASE_DATASET_DIRECTORY}")
    log.info(f"Aligned master raster shape: {radiation.shape}")
    log.info(
        "Radiation dose rate: "
        f"min={radiation.min():.4f}, "
        f"max={radiation.max():.4f} {metadata['unit']}"
    )
    log.info(
        "Resolved scenario: "
        f"{metadata['source_type']} / {metadata['shape_type']} / "
        f"{metadata['intensity_type']}; seed={metadata['seed']}"
    )
    return result


if __name__ == "__main__":
    run_radiation_layer_example()
