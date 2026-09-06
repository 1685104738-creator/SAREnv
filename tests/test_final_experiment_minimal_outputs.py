from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from sarenv.evaluation import PostRunEvaluationResult, RadiationQuantityContract


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "radiation"
    / "09_final_experiment.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("final_experiment_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _evaluation_result(factor: float) -> PostRunEvaluationResult:
    summary = {
        "sar": {
            "total_likelihood_score": 0.5 * factor,
            "total_time_discounted_score": 0.25 * factor,
            "area_covered_km2": factor,
        },
        "uav_radiation": {
            "uav_excess_exposure_distance_integral": 10.0 * factor,
            "uav_total_exposure_distance_integral": 12.0 * factor,
            "uav_cumulative_excess_dose": 0.1 * factor,
            "uav_cumulative_total_dose": 0.12 * factor,
        },
        "survivor_radiation": {
            "found_count": int(10 * factor),
            "all_survivors": {
                "pre_discovery_total_dose": {
                    "sum": 20.0 * factor,
                    "mean": 2.0 * factor,
                    "median": 1.5 * factor,
                    "p95": 4.0 * factor,
                    "max": 5.0 * factor,
                }
            },
        },
        "trajectory": {
            "total_distance_m": 100.0 * factor,
            "turn_diagnostics": {
                "number_of_heading_changes": int(3 * factor),
                "total_absolute_heading_change_deg": 90.0 * factor,
            },
        },
    }
    return PostRunEvaluationResult(
        evaluation_summary=summary,
        sar_metrics=summary["sar"],
        uav_radiation_summary=summary["uav_radiation"],
        survivor_radiation_summary=summary["survivor_radiation"],
        output_paths={},
    )


def _scenario(runner, directory_name: str, scenario_id: str, scenario_type: str):
    return runner.FrozenRadiationScenario(
        directory_name=directory_name,
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        measurement_quantity="test_rate",
        contract=RadiationQuantityContract("test_rate", "u/h", "u"),
        background_rate=0.0,
        truth_query_excess=lambda x, y, z: 0.0,
        truth_query_total=lambda x, y, z: 0.0,
        source_positions_m=((100.0, 200.0),),
        config={},
    )


def test_intensity_sweep_is_minimal_resumable_and_reuses_original(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    output_root = tmp_path / "intensity"
    context = SimpleNamespace(max_radius_m=600.0, search_center=(100.0, 200.0))
    monkeypatch.setattr(runner, "load_frozen_context", lambda: context)
    monkeypatch.setattr(runner, "_trial_context", lambda base, seed: base)
    monkeypatch.setattr(runner, "_trial_anchors", lambda base, seed: ((0, 0),) * 3)

    baseline_calls = []

    def fake_baseline(base, *, minimal_output=False):
        assert minimal_output
        baseline_calls.append(base)
        path = runner.FINAL_ROOT / "baseline" / "mission_steps.csv"
        runner._csv(path, [{"step_index": 0}])
        return path

    monkeypatch.setattr(runner, "generate_original_baseline", fake_baseline)
    monkeypatch.setattr(runner, "load_executed_trajectory", lambda path: object())
    route_samples = object()
    monkeypatch.setattr(
        runner,
        "prepare_route_integration_samples",
        lambda trajectory, *, integration_step_m: route_samples,
    )

    scenarios = (
        _scenario(runner, "single", "single_point_formal_first_run", "point_only"),
        _scenario(runner, "multi", "final_three_point_stress_scenario", "multi_point"),
        _scenario(runner, "surface", "final_uniform_cs137_surface", "surface_only"),
    )
    frozen_intensities = []

    def fake_freeze(base, anchors, intensity):
        frozen_intensities.append(intensity)
        return scenarios

    monkeypatch.setattr(runner, "_freeze_intensity_scenarios", fake_freeze)

    def fake_mission(base, scenario, *, minimal_output=False):
        assert minimal_output
        path = runner.FINAL_ROOT / scenario.directory_name / "mission_steps.csv"
        runner._csv(path, [{"step_index": 0}])
        return runner.RadiationAwareMissionResult(
            trajectory_path=path,
            summary={
                "status": "COMPLETE",
                "radiation_episodes": 1,
                "radiation_steps": 2,
                "radiation_priority_decisions": 3,
            },
        )

    monkeypatch.setattr(runner, "run_radiation_aware_mission", fake_mission)

    def fake_evaluate(base, scenario, *, planner_name, **kwargs):
        if planner_name == "original":
            assert kwargs["route_integration_samples"] is route_samples
            return _evaluation_result(1.0)
        return _evaluation_result(2.0)

    monkeypatch.setattr(runner, "evaluate_route", fake_evaluate)
    runner.run_intensity_repeated_trials(
        (runner.TRIAL_SEEDS[0],), (0.25, 0.5), output_root=output_root
    )

    assert len(baseline_calls) == 1
    assert frozen_intensities == [0.25, 0.5]
    status = runner._read_optional_csv(output_root / "run_status.csv")
    summary = runner._read_optional_csv(output_root / "intensity_summary.csv")
    assert [row["status"] for row in status] == ["SUCCESS", "SUCCESS"]
    assert len(summary) == 6
    assert {float(row["Y"]) for row in summary} == {0.05}
    assert {float(row["sigma_m"]) for row in summary} == {40.0}
    assert {float(row["influence_radius_m"]) for row in summary} == {120.0}
    assert sorted(path.name for path in output_root.iterdir()) == [
        "experiment_config.json",
        "intensity_summary.csv",
        "run_status.csv",
        "sweep.log",
    ]

    runner.run_intensity_repeated_trials(
        (runner.TRIAL_SEEDS[0],), (0.25, 0.5), output_root=output_root
    )
    assert len(baseline_calls) == 1
    assert len(runner._read_optional_csv(output_root / "intensity_summary.csv")) == 6


def test_minimal_evaluation_disables_all_evaluator_outputs(monkeypatch, tmp_path):
    runner = _load_runner()
    captured = {}
    sentinel = _evaluation_result(1.0)
    route_samples = object()

    class FakeEvaluator:
        def __init__(self, config, **kwargs):
            captured["config"] = config
            captured["route_integration_samples"] = kwargs["route_integration_samples"]

        def evaluate(self):
            return sentinel

    monkeypatch.setattr(runner, "PostRunEvaluator", FakeEvaluator)
    context = SimpleNamespace(survivors_path=tmp_path / "survivors.json")
    frozen = _scenario(runner, "single", "single", "point_only")
    result = runner.evaluate_route(
        context,
        frozen,
        planner_name="original",
        trajectory_path=tmp_path / "mission_steps.csv",
        minimal_output=True,
        route_integration_samples=route_samples,
    )
    assert result is sentinel
    assert not captured["config"].write_detail_outputs
    assert not captured["config"].write_cumulative_metrics
    assert not captured["config"].generate_figures
    assert captured["route_integration_samples"] is route_samples


def test_formal_defaults_and_point_intensity_scale_only_source(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setattr(runner, "FINAL_ROOT", tmp_path)
    assert runner.TRIAL_SEEDS == (
        113, 227, 331, 439, 547, 653, 761, 877, 983, 1091,
        1193, 1297, 1423, 1523, 1627, 1733, 1831, 1931, 2027, 2131,
    )
    assert runner.RadiationPlannerParameters() == runner.RadiationPlannerParameters(
        hazard_reference_y=0.05,
        kernel_sigma_m=40.0,
        influence_radius_m=120.0,
    )
    context = SimpleNamespace(
        search_center=(500000.0, 5700000.0),
        item=SimpleNamespace(projected_crs="EPSG:32630", meter_per_bin=20.0),
        bounds=(499400.0, 5699400.0, 500600.0, 5700600.0),
        probability_map=np.full((60, 60), 1.0 / 3600.0),
        detection_radius_m=20.71067811865475,
        mission_step_allowance=3600,
        heatmap_sha256="heatmap",
        survivor_sha256="survivors",
    )
    scenario = runner.freeze_point_scenario(
        context, multi=True, intensity_multiplier=4.0
    )
    strengths = {
        float(source["reference_excess_uSv_h"])
        for source in scenario.config["sources"]
    }
    assert strengths == {4.0 * runner.POINT_REFERENCE_EXCESS_USV_H}
    assert len(scenario.config["sources"]) == 3
    assert scenario.background_rate == runner.POINT_BACKGROUND_USV_H
    assert scenario.config["background_rate"] == runner.POINT_BACKGROUND_USV_H
    assert scenario.config["source_intensity_multiplier"] == 4.0
