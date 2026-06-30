import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from argument_correlation_llm import DEFAULT_DISTANCE_FILE, calculate_llm_argument_correlation
from correlation_calculator import calculate_period_correlations
from data_representation import TruckDayLookup, _replace_none_with_nan
from image_cluster import cluster_vehicle_images
from main import (
    DEFAULT_ARGUMENT_WEIGHTS,
    DEFAULT_OUTPUT_DIR,
    _date_range,
    _normalize_date,
    _normalize_weights,
    _route_appearance_rows,
    _sample_final_scenario_days,
    _target_pair_rows,
    _target_route_rows,
    _truck_appearance_frequencies,
    _write_json,
)


DEFAULT_LLM_OUTPUT_DIR = Path("scenario_generation_llm")


def build_llm_scenario_generation(
    target_vehicle: int,
    other_vehicles: list[int],
    start_date: str,
    end_date: str,
    output_dir: Path = DEFAULT_LLM_OUTPUT_DIR,
    image_feature_method: str = "pca",
    image_clusters: int = 10,
    scenario_days: int = 2,
    seed: int = 0,
    run_dir: Optional[Path] = None,
    argument_weights: Optional[dict[str, float]] = None,
    distance_file: Path = DEFAULT_DISTANCE_FILE,
    llm_provider: str = "anthropic",
    llm_model: Optional[str] = None,
) -> dict:
    normalized_start = _normalize_date(start_date)
    normalized_end = _normalize_date(end_date)
    if normalized_start > normalized_end:
        raise ValueError("--start-date must be before or equal to --end-date.")

    vehicles = [target_vehicle, *other_vehicles]
    period_dates = _date_range(normalized_start, normalized_end)
    period_day_count = len(period_dates)

    run_label = (
        f"target_{target_vehicle}_others_{'_'.join(str(vehicle_id) for vehicle_id in other_vehicles)}_"
        f"{normalized_start}_{normalized_end}_llm"
    )
    run_dir = run_dir or output_dir / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    lookup = TruckDayLookup()
    data_representations = {}
    for vehicle_id in vehicles:
        records = [
            _replace_none_with_nan(asdict(lookup.get(vehicle_id, date)))
            for date in period_dates
        ]
        available_route_days = sum(1 for record in records if not isinstance(record["gps_image"], float))
        data_representations[str(vehicle_id)] = {
            "vehicle_id": vehicle_id,
            "period_days": period_day_count,
            "available_route_days": available_route_days,
            "records": records,
        }
    _write_json(run_dir / "data_representations.json", data_representations)

    stop_profile_correlation = calculate_period_correlations(
        passes_file=Path("vehicle_day_zone_passes.csv"),
        vehicle_ids=vehicles,
        start_date=normalized_start,
        end_date=normalized_end,
    )
    _write_json(run_dir / "stop_profile_correlations.json", stop_profile_correlation)

    image_clusters_by_vehicle = {}
    image_cluster_dir = run_dir / "image_cluster"
    for vehicle_id in vehicles:
        image_clusters_by_vehicle[str(vehicle_id)] = cluster_vehicle_images(
            vehicle_id=vehicle_id,
            start_date=normalized_start,
            end_date=normalized_end,
            output_dir=image_cluster_dir,
            feature_method=image_feature_method,
            n_clusters=image_clusters,
        )

    argument_correlation = calculate_llm_argument_correlation(
        vehicle_ids=vehicles,
        start_date=normalized_start,
        end_date=normalized_end,
        weights=_normalize_weights(argument_weights or DEFAULT_ARGUMENT_WEIGHTS),
        image_cluster_results=image_clusters_by_vehicle,
        route_cluster_output_dir=run_dir / "argument_route_clusters",
        route_feature_method=image_feature_method,
        route_clusters=image_clusters,
        distance_file=distance_file,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    _write_json(run_dir / "argument_correlations_llm.json", argument_correlation)

    argument_pairs = _target_pair_rows(
        argument_correlation["pairwise_argument_correlations"],
        target_vehicle,
    )
    truck_appearance = _truck_appearance_frequencies(argument_pairs, other_vehicles)
    truck_appearance = {
        vehicle_id: {
            **values,
            "source": "normalized_positive_llm_reasoned_argument_correlation",
        }
        for vehicle_id, values in truck_appearance.items()
    }

    target_cluster_result = image_clusters_by_vehicle[str(target_vehicle)]
    target_route_appearances = _target_route_rows(
        target_vehicle,
        target_cluster_result,
        period_day_count,
    )

    route_appearances = []
    for vehicle_id in other_vehicles:
        cluster_result = image_clusters_by_vehicle[str(vehicle_id)]
        route_appearances.extend(
            _route_appearance_rows(
                vehicle_id,
                cluster_result,
                period_day_count,
                truck_appearance[vehicle_id]["appearance_frequency"],
            )
        )
    route_appearances.sort(key=lambda row: row["joint_scenario_frequency"], reverse=True)

    final_scenario = _sample_final_scenario_days(
        target_route_appearances,
        route_appearances,
        scenario_days,
        seed,
    )

    _write_json(run_dir / "image_clusters.json", image_clusters_by_vehicle)
    _write_json(
        run_dir / "appearance_frequencies_llm.json",
        {
            "truck_appearance_frequencies": {
                str(vehicle_id): values
                for vehicle_id, values in truck_appearance.items()
            },
            "target_route_appearance_frequencies": target_route_appearances,
            "route_appearance_frequencies": route_appearances,
        },
    )
    _write_json(run_dir / "final_llm_scenario.json", final_scenario)

    result = {
        "inputs": {
            "target_vehicle": target_vehicle,
            "other_vehicles": other_vehicles,
            "vehicles": vehicles,
            "start_date": normalized_start,
            "end_date": normalized_end,
            "period_days": period_day_count,
            "image_feature_method": image_feature_method,
            "image_clusters": image_clusters,
            "scenario_days": scenario_days,
            "seed": seed,
            "argument_weights": _normalize_weights(argument_weights or DEFAULT_ARGUMENT_WEIGHTS),
            "distance_file": str(distance_file),
            "llm_provider": llm_provider,
            "llm_model": argument_correlation["method"].get("llm_model"),
        },
        "method": {
            "step_1": "Represent each truck-day with stop-zone, GPS image, route distance, and fleet metadata.",
            "step_2": "Build stop-profile vectors and route-image clusters as evidence.",
            "step_3": "Ask the LLM to reason over statistical evidence and produce final pairwise truck correlations.",
            "step_4": "Normalize positive LLM-reasoned target-to-other correlations into truck appearance frequencies.",
            "step_5": "Sample target route clusters by target route frequency and other-truck routes by joint frequency.",
            "truck_appearance_frequency": "Positive target-to-other LLM-reasoned correlations normalized across other trucks.",
            "route_appearance_frequency": "Cluster days divided by selected period length.",
            "joint_scenario_frequency": "LLM truck appearance frequency multiplied by route appearance frequency.",
        },
        "target_pair_argument_correlations": {
            str(vehicle_id): argument_pairs.get(vehicle_id)
            for vehicle_id in other_vehicles
        },
        "truck_appearance_frequencies": {
            str(vehicle_id): values
            for vehicle_id, values in truck_appearance.items()
        },
        "target_route_appearance_frequencies": target_route_appearances,
        "route_appearance_frequencies": route_appearances,
        "final_scenario": final_scenario,
        "output_files": {
            "summary": str(run_dir / "scenario_summary_llm.json"),
            "data_representations": str(run_dir / "data_representations.json"),
            "stop_profile_correlations": str(run_dir / "stop_profile_correlations.json"),
            "argument_correlations": str(run_dir / "argument_correlations_llm.json"),
            "image_clusters": str(run_dir / "image_clusters.json"),
            "appearance_frequencies": str(run_dir / "appearance_frequencies_llm.json"),
            "final_scenario": str(run_dir / "final_llm_scenario.json"),
        },
    }
    _write_json(run_dir / "scenario_summary_llm.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run LLM-guided scenario generation: statistical evidence is computed first, "
            "then an LLM reasons pairwise correlations used for sampling."
        )
    )
    parser.add_argument("--target-vehicle", type=int, required=True)
    parser.add_argument("--other-vehicles", nargs="+", type=int, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LLM_OUTPUT_DIR)
    parser.add_argument("--image-feature-method", choices=("pca", "autoencoder"), default="pca")
    parser.add_argument("--image-clusters", type=int, default=10)
    parser.add_argument("--scenario-days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance-file", type=Path, default=DEFAULT_DISTANCE_FILE)
    parser.add_argument("--llm-provider", choices=("anthropic", "openai", "deepseek", "glm"), default="anthropic")
    parser.add_argument("--llm-model")
    args = parser.parse_args()

    result = build_llm_scenario_generation(
        target_vehicle=args.target_vehicle,
        other_vehicles=args.other_vehicles,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        image_feature_method=args.image_feature_method,
        image_clusters=args.image_clusters,
        scenario_days=args.scenario_days,
        seed=args.seed,
        distance_file=args.distance_file,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    print(f"Wrote LLM scenario summary to {result['output_files']['summary']}")
    print("LLM truck appearance frequencies:")
    for vehicle_id, values in result["truck_appearance_frequencies"].items():
        print(
            f"  Vehicle {vehicle_id}: "
            f"llm_correlation={values['raw_total_correlation']:.3f}, "
            f"appearance_frequency={values['appearance_frequency']:.3f}"
        )
    print(f"Final LLM scenario written to {result['output_files']['final_scenario']}")
    for day in result["final_scenario"]["days"]:
        target = day["target_vehicle"]
        other = day["other_vehicle"]
        print(
            f"  Day {day['scenario_day']}: "
            f"target Vehicle {target['vehicle_id']} cluster {target['cluster']} "
            f"sampled_route_date={target['sampled_route_date']}; "
            f"other Vehicle {other['vehicle_id']} cluster {other['cluster']} "
            f"sampled_route_date={other['sampled_route_date']}"
        )


if __name__ == "__main__":
    main()
