"""Evaluate the saved adaptive-entry 0.01x mission without rerunning its planner."""

from __future__ import annotations

import json
from pathlib import Path

from sarenv.evaluation import EvaluationConfig, PostRunEvaluator
from sarenv.radiation import PointSource, PointSourceConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MISSION_DIRECTORY = (
    REPOSITORY_ROOT
    / "results"
    / "radiation_aware_adaptive_entry"
    / "initial_entry_0p01x"
)
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "results" / "evaluation" / "adaptive_entry_0p01x"
)

# No authoritative cruise speed exists in the current experiment configuration.
EVALUATION_SPEED_M_S = 10.0
SPEED_STATUS = "provisional_for_evaluator_test"
RADIATION_INTEGRATION_STEP_M = 1.0


def _point_source_from_parameters(parameters: dict[str, object]) -> PointSource:
    metadata = parameters["point_source_truth_debug_evaluation_only"]
    if not isinstance(metadata, dict) or metadata.get("scenario_type") != "point_only":
        raise ValueError("This smoke evaluator requires a saved point-only truth.")
    return PointSource(
        PointSourceConfig(
            source_id=str(metadata["source_id"]),
            x_m=float(metadata["x_m"]),
            y_m=float(metadata["y_m"]),
            source_height_m=float(metadata["source_height_m"]),
            reference_distance_m=float(metadata["reference_distance_m"]),
            reference_excess_uSv_h=float(metadata["reference_excess_uSv_h"]),
            core_radius_m=float(metadata["core_radius_m"]),
            cutoff_radius_m=(
                None
                if metadata["cutoff_radius_m"] is None
                else float(metadata["cutoff_radius_m"])
            ),
            crs=str(metadata["crs"]),
            unit=str(metadata["unit"]),
            model_name=str(metadata["model_name"]),
        )
    )


def main() -> None:
    parameters_path = MISSION_DIRECTORY / "resolved_parameters.json"
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    point_source = _point_source_from_parameters(parameters)
    dataset_directory = Path(str(parameters["dataset_directory"]))
    evaluator = PostRunEvaluator(
        EvaluationConfig(
            trajectory_path=MISSION_DIRECTORY / "mission_steps.csv",
            dataset_directory=dataset_directory,
            survivors_path=dataset_directory / "lost_persons.json",
            scenario_parameters_path=parameters_path,
            output_directory=OUTPUT_DIRECTORY,
            platform_altitude_m=float(parameters["platform_altitude_m"]),
            survivor_reference_height_m=float(parameters["value_reference_height_m"]),
            fov_deg=float(parameters["fov_deg"]),
            background_uSv_h=float(parameters["background_uSv_h"]),
            radiation_integration_step_m=RADIATION_INTEGRATION_STEP_M,
            evaluation_speed_m_s=EVALUATION_SPEED_M_S,
            speed_status=SPEED_STATUS,
        ),
        excess_truth_query=point_source.query_excess_dose_rate,
        source_position_m=(point_source.config.x_m, point_source.config.y_m),
    )
    result = evaluator.evaluate()
    print(json.dumps(result.evaluation_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
