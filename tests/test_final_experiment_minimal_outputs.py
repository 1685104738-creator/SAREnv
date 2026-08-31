from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from shapely.geometry import Point

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


def _evaluation_summary(factor: float) -> dict[str, object]:
    return {
        "sar": {
            "total_likelihood_score": 0.5 * factor,
            "total_time_discounted_score": 0.25 * factor,
            "area_covered_km2": 1.0 * factor,
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


def _evaluation_result(factor: float) -> PostRunEvaluationResult:
    summary = _evaluation_summary(factor)
    return PostRunEvaluationResult(
        evaluation_summary=summary,
        sar_metrics=summary["sar"],
        uav_radiation_summary=summary["uav_radiation"],
        survivor_radiation_summary=summary["survivor_radiation"],
        output_paths={},
    )


def test_repeated_trial_writes_only_minimal_output_set(tmp_path, monkeypatch):
    runner = _load_runner()
    repeated_root = tmp_path / "repeated"
    monkeypatch.setattr(runner, "REPEATED_ROOT", repeated_root)
    context = SimpleNamespace(
        survivors=SimpleNamespace(points=(Point(0.0, 0.0),)),
        max_radius_m=600.0,
        search_center=(100.0, 200.0),
    )
    monkeypatch.setattr(runner, "load_frozen_context", lambda: context)

    def fake_trial_context(base, trial_seed):
        assert trial_seed in {203, 307}
        runner._json(runner.FINAL_ROOT / "inputs" / "lost_persons.json", {})
        return base

    monkeypatch.setattr(runner, "_trial_context", fake_trial_context)
    monkeypatch.setattr(
        runner,
        "_sample_legal_anchor",
        lambda rng, offsets, radius_m: (0.0, 0.0),
    )

    def fake_baseline(base, *, minimal_output=False):
        assert minimal_output
        path = (
            runner.FINAL_ROOT
            / "baseline_original"
            / "simulation"
            / "mission_steps.csv"
        )
        runner._csv(path, [{"step_index": 0}])
        return path

    monkeypatch.setattr(runner, "generate_original_baseline", fake_baseline)
    original_samples = object()
    monkeypatch.setattr(runner, "load_executed_trajectory", lambda path: object())
    prepare_calls = []

    def fake_prepare(trajectory, *, integration_step_m):
        prepare_calls.append((trajectory, integration_step_m))
        return original_samples

    monkeypatch.setattr(runner, "prepare_route_integration_samples", fake_prepare)

    def scenario(directory_name, scenario_id, scenario_type):
        runner._json(runner.FINAL_ROOT / directory_name / "scenario_config.json", {})
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

    monkeypatch.setattr(
        runner,
        "freeze_point_scenario",
        lambda base, *, multi, anchor_offset_m=None: scenario(
            "02_multi_point" if multi else "01_single_point",
            "multi" if multi else "single",
            "multi_point" if multi else "point_only",
        ),
    )

    def fake_surface(base, *, anchor_offset_m=(0.0, 0.0), save_truth_outputs=True):
        assert not save_truth_outputs
        return scenario("03_uniform_surface", "surface", "surface_only")

    monkeypatch.setattr(runner, "freeze_surface_scenario", fake_surface)

    def fake_run_one(
        base,
        frozen,
        baseline_path,
        *,
        minimal_output=False,
        original_route_integration_samples=None,
    ):
        assert minimal_output
        assert original_route_integration_samples is original_samples
        path = (
            runner.FINAL_ROOT
            / frozen.directory_name
            / "radiation_aware"
            / "simulation"
            / "mission_steps.csv"
        )
        runner._csv(path, [{"step_index": 0}])
        mission = runner.RadiationAwareMissionResult(
            trajectory_path=path,
            summary={
                "radiation_episodes": 1,
                "radiation_steps": 2,
                "radiation_priority_decisions": 3,
            },
        )
        return runner.ScenarioRunResult(
            scenario=frozen,
            aware_mission=mission,
            original_evaluation=_evaluation_result(1.0),
            aware_evaluation=_evaluation_result(2.0),
        )

    monkeypatch.setattr(runner, "run_one_scenario", fake_run_one)

    def fake_figure(base, frozen, aware_path, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")

    monkeypatch.setattr(runner, "_minimal_trajectory_comparison_figure", fake_figure)

    runner.run_spatial_repeated_trials((203, 307), trial_id_start=2)

    assert len(prepare_calls) == 1
    expected_files = sorted(
        [
            "inputs/lost_persons.json",
            "baseline_original/simulation/mission_steps.csv",
            "01_single_point/scenario_config.json",
            "01_single_point/radiation_aware/simulation/mission_steps.csv",
            "02_multi_point/scenario_config.json",
            "02_multi_point/radiation_aware/simulation/mission_steps.csv",
            "03_uniform_surface/scenario_config.json",
            "03_uniform_surface/radiation_aware/simulation/mission_steps.csv",
            "combined/final_summary.csv",
            "combined/figures/single_trajectory_comparison.png",
            "combined/figures/multi_trajectory_comparison.png",
            "combined/figures/surface_trajectory_comparison.png",
        ]
    )
    for trial_root in (
        repeated_root / "trial_02_seed_203",
        repeated_root / "trial_03_seed_307",
    ):
        relative_files = sorted(
            path.relative_to(trial_root).as_posix()
            for path in trial_root.rglob("*")
            if path.is_file()
        )
        assert relative_files == expected_files
    assert (repeated_root / "repeated_trials_summary.csv").exists()


def test_minimal_evaluation_disables_all_evaluator_outputs(monkeypatch, tmp_path):
    runner = _load_runner()
    captured = {}
    sentinel = _evaluation_result(1.0)
    route_samples = object()

    class FakeEvaluator:
        def __init__(self, config, **kwargs):
            captured["config"] = config
            captured["route_integration_samples"] = kwargs[
                "route_integration_samples"
            ]

        def evaluate(self):
            return sentinel

    monkeypatch.setattr(runner, "PostRunEvaluator", FakeEvaluator)
    context = SimpleNamespace(survivors_path=tmp_path / "survivors.json")
    frozen = runner.FrozenRadiationScenario(
        directory_name="01_single_point",
        scenario_id="single",
        scenario_type="point_only",
        measurement_quantity="test_rate",
        contract=RadiationQuantityContract("test_rate", "u/h", "u"),
        background_rate=0.0,
        truth_query_excess=lambda x, y, z: 0.0,
        truth_query_total=lambda x, y, z: 0.0,
        source_positions_m=(),
        config={},
    )

    result = runner.evaluate_route(
        context,
        frozen,
        planner_name="original",
        trajectory_path=tmp_path / "mission_steps.csv",
        minimal_output=True,
        route_integration_samples=route_samples,
    )

    assert result is sentinel
    config = captured["config"]
    assert not config.write_detail_outputs
    assert not config.write_cumulative_metrics
    assert not config.generate_figures
    assert captured["route_integration_samples"] is route_samples
