import json
from pathlib import Path

from main import _validate_scenario_distances, build_scenario_generation


TARGET_VEHICLE = 155
SEEDS = [1, 2, 3, 4, 5]
VALIDATION_DATES = [
    "2023-01-05",
    "2023-01-15",
    "2023-01-15",
    "2023-01-31",
    "2023-01-25",
]
CASES = {
    5: [1181, 689, 1236, 1206],
    10: [1247, 1311, 1312, 1350, 1363, 1421, 1423, 1506, 1575],
    20: [
        1247,
        1311,
        1312,
        1350,
        1363,
        1421,
        1423,
        1506,
        1575,
        1576,
        1578,
        1667,
        1687,
        1688,
        1785,
        1993,
        2191,
        2201,
        2518,
    ],
}


def main() -> None:
    output_root = Path("gpt_5_5_global_seed_5_non_llm_benchmarks")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for vehicle_count, other_vehicles in CASES.items():
        seed_runs = []
        for seed in SEEDS:
            run_dir = output_root / f"{vehicle_count}_vehicles" / f"seed_{seed}"
            scenario = build_scenario_generation(
                target_vehicle=TARGET_VEHICLE,
                other_vehicles=other_vehicles,
                start_date="2023-01-01",
                end_date="2023-01-31",
                output_dir=output_root,
                image_feature_method="pca",
                image_clusters=5,
                scenario_days=5,
                seed=seed,
                run_dir=run_dir,
            )
            validation = _validate_scenario_distances(
                scenario,
                VALIDATION_DATES,
                [TARGET_VEHICLE, *other_vehicles],
            )
            seed_run = {
                "seed": seed,
                "scenario_summary": scenario["output_files"]["summary"],
                "final_scenario": scenario["output_files"]["final_scenario"],
                "mean_individual_relative_distance_error": validation[
                    "mean_individual_relative_distance_error"
                ],
                "fleet_total_relative_distance_error": validation[
                    "fleet_total_relative_distance_error"
                ],
                "validation": validation,
            }
            seed_runs.append(seed_run)
            (run_dir / "validation_corrected.json").write_text(
                json.dumps(validation, indent=2) + "\n",
                encoding="utf-8",
            )

        mean_individual_values = [
            seed_run["mean_individual_relative_distance_error"]
            for seed_run in seed_runs
            if seed_run["mean_individual_relative_distance_error"] is not None
        ]
        fleet_values = [
            seed_run["fleet_total_relative_distance_error"]
            for seed_run in seed_runs
            if seed_run["fleet_total_relative_distance_error"] is not None
        ]
        row = {
            "vehicle_count": vehicle_count,
            "target_vehicle": TARGET_VEHICLE,
            "other_vehicles": other_vehicles,
            "seeds": SEEDS,
            "scenario_days": 5,
            "image_clusters": 5,
            "seed_runs": seed_runs,
            "mean_individual_relative_distance_error": (
                sum(mean_individual_values) / len(mean_individual_values)
                if mean_individual_values
                else None
            ),
            "fleet_total_relative_distance_error": (
                sum(fleet_values) / len(fleet_values)
                if fleet_values
                else None
            ),
        }
        rows.append(row)

    summary = {
        "description": (
            "Corrected non-LLM statistical baselines for the GPT 5.5 vehicle "
            "sets across the same five sampling seeds used by the LLM "
            "experiments. Non-sampled vehicles are excluded from error "
            "calculation rather than counted as generated distance 0."
        ),
        "seeds": SEEDS,
        "validation_dates": VALIDATION_DATES,
        "runs": rows,
    }
    summary_path = output_root / "non_llm_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    for row in rows:
        print(
            f"{row['vehicle_count']} vehicles | "
            f"mean={row['mean_individual_relative_distance_error']:.6f} | "
            f"fleet={row['fleet_total_relative_distance_error']:.6f}"
        )


if __name__ == "__main__":
    main()
