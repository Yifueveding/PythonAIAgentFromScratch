import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from argument_correlation import calculate_statistical_argument_correlation
from correlation_calculator import calculate_period_correlations
from data_representation import TruckDayLookup, _replace_none_with_nan
from image_cluster import cluster_vehicle_images


DEFAULT_OUTPUT_DIR = Path("scenario_generation")
DEFAULT_ARGUMENT_WEIGHTS = {
    "stop_profile_correlation": 0.45,
    "route_similarity": 0.15,
    "purpose_match": 0.12,
    "age_similarity": 0.08,
    "kms_similarity": 0.08,
    "duty_cycle_match": 0.12,
}


def _normalize_date(value: str) -> str:
    for date_format in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value[:10], date_format).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value[:10]


def _date_range(start_date: str, end_date: str) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = []
    current = start
    while current <= end:
        days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return days


def _target_pair_rows(rows: list[dict], target_vehicle: int) -> dict[int, dict]:
    pairs = {}
    for row in rows:
        if row["vehicle_a"] == target_vehicle:
            pairs[row["vehicle_b"]] = row
        elif row["vehicle_b"] == target_vehicle:
            pairs[row["vehicle_a"]] = row
    return pairs


def _positive_score(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return max(0.0, float(value))


def _truck_appearance_frequencies(argument_pairs: dict[int, dict], other_vehicles: list[int]) -> dict[int, dict]:
    raw_scores = {
        vehicle_id: _positive_score(argument_pairs.get(vehicle_id, {}).get("total_correlation"))
        for vehicle_id in other_vehicles
    }
    total = sum(raw_scores.values())
    if total == 0:
        frequencies = {vehicle_id: 0.0 for vehicle_id in other_vehicles}
    else:
        frequencies = {
            vehicle_id: score / total
            for vehicle_id, score in raw_scores.items()
        }

    return {
        vehicle_id: {
            "raw_total_correlation": raw_scores[vehicle_id],
            "appearance_frequency": frequencies[vehicle_id],
            "source": "normalized_positive_statistical_argument_correlation",
        }
        for vehicle_id in other_vehicles
    }


def _route_appearance_rows(
    vehicle_id: int,
    image_cluster_result: dict,
    period_day_count: int,
    truck_appearance_frequency: float,
) -> list[dict]:
    rows = []
    for cluster in image_cluster_result["cluster_summary"]:
        route_frequency = cluster["days"] / period_day_count if period_day_count else 0.0
        rows.append(
            {
                "vehicle_id": vehicle_id,
                "cluster": cluster["cluster"],
                "route_days": cluster["days"],
                "period_days": period_day_count,
                "route_appearance_frequency": route_frequency,
                "truck_appearance_frequency": truck_appearance_frequency,
                "joint_scenario_frequency": truck_appearance_frequency * route_frequency,
                "representative_date": cluster["representative_date"],
                "example_dates": cluster["example_dates"],
            }
        )
    rows.sort(key=lambda row: row["joint_scenario_frequency"], reverse=True)
    return rows


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _build_image_cluster_cache(
    vehicle_ids: list[int],
    start_date: str,
    end_date: str,
    output_dir: Path,
    image_feature_method: str,
    image_clusters: int,
) -> dict[str, dict]:
    cache = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for vehicle_id in vehicle_ids:
        cache[str(vehicle_id)] = cluster_vehicle_images(
            vehicle_id=vehicle_id,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            feature_method=image_feature_method,
            n_clusters=image_clusters,
        )
    return cache


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.0, float(weights.get(key, 0.0))) for key in DEFAULT_ARGUMENT_WEIGHTS}
    total = sum(cleaned.values())
    if total == 0:
        return dict(DEFAULT_ARGUMENT_WEIGHTS)
    return {key: value / total for key, value in cleaned.items()}


def _sample_weighted(rows: list[dict], weight_key: str, rng: random.Random) -> dict:
    if not rows:
        raise ValueError("Cannot sample from an empty row list.")

    weights = [max(0.0, float(row.get(weight_key) or 0.0)) for row in rows]
    total = sum(weights)
    if total == 0:
        return rng.choice(rows)

    threshold = rng.random() * total
    running_total = 0.0
    for row, weight in zip(rows, weights):
        running_total += weight
        if running_total >= threshold:
            return row
    return rows[-1]


def _target_route_rows(
    vehicle_id: int,
    image_cluster_result: dict,
    period_day_count: int,
) -> list[dict]:
    rows = []
    for cluster in image_cluster_result["cluster_summary"]:
        route_frequency = cluster["days"] / period_day_count if period_day_count else 0.0
        rows.append(
            {
                "vehicle_id": vehicle_id,
                "cluster": cluster["cluster"],
                "route_days": cluster["days"],
                "period_days": period_day_count,
                "route_appearance_frequency": route_frequency,
                "representative_date": cluster["representative_date"],
                "example_dates": cluster["example_dates"],
            }
        )
    rows.sort(key=lambda row: row["route_appearance_frequency"], reverse=True)
    return rows


def _sample_final_scenario_days(
    target_route_appearances: list[dict],
    route_appearances: list[dict],
    scenario_days: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    sampled_days = []
    for day_index in range(1, scenario_days + 1):
        sampled_target_route = _sample_weighted(
            target_route_appearances,
            "route_appearance_frequency",
            rng,
        )
        target_example_dates = sampled_target_route.get("example_dates") or []
        sampled_target_date = (
            rng.choice(target_example_dates)
            if target_example_dates
            else sampled_target_route.get("representative_date")
        )

        sampled_other_route = _sample_weighted(route_appearances, "joint_scenario_frequency", rng)
        other_example_dates = sampled_other_route.get("example_dates") or []
        sampled_other_date = (
            rng.choice(other_example_dates)
            if other_example_dates
            else sampled_other_route.get("representative_date")
        )
        sampled_days.append(
            {
                "scenario_day": day_index,
                "target_vehicle": {
                    "vehicle_id": sampled_target_route["vehicle_id"],
                    "cluster": sampled_target_route["cluster"],
                    "sampled_route_date": sampled_target_date,
                    "representative_date": sampled_target_route.get("representative_date"),
                    "route_appearance_frequency": sampled_target_route["route_appearance_frequency"],
                    "sampling_weight": "route_appearance_frequency",
                },
                "other_vehicle": {
                    "vehicle_id": sampled_other_route["vehicle_id"],
                    "cluster": sampled_other_route["cluster"],
                    "sampled_route_date": sampled_other_date,
                    "representative_date": sampled_other_route.get("representative_date"),
                    "truck_appearance_frequency": sampled_other_route["truck_appearance_frequency"],
                    "route_appearance_frequency": sampled_other_route["route_appearance_frequency"],
                    "joint_scenario_frequency": sampled_other_route["joint_scenario_frequency"],
                    "sampling_weight": "joint_scenario_frequency",
                },
            }
        )

    return {
        "scenario_days_requested": scenario_days,
        "seed": seed,
        "sampling_method": (
            "For each scenario day, sample the target truck route from its own "
            "route_appearance_frequency, then sample one other truck route from "
            "joint_scenario_frequency = truck_appearance_frequency * route_appearance_frequency."
        ),
        "days": sampled_days,
    }


def build_scenario_generation(
    target_vehicle: int,
    other_vehicles: list[int],
    start_date: str,
    end_date: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    image_feature_method: str = "pca",
    image_clusters: int = 10,
    scenario_days: int = 2,
    seed: int = 0,
    run_dir: Optional[Path] = None,
    argument_weights: Optional[dict[str, float]] = None,
    precomputed_image_clusters: Optional[dict[str, dict]] = None,
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
        f"{normalized_start}_{normalized_end}"
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
        if precomputed_image_clusters and str(vehicle_id) in precomputed_image_clusters:
            cluster_result = precomputed_image_clusters[str(vehicle_id)]
        else:
            cluster_result = cluster_vehicle_images(
                vehicle_id=vehicle_id,
                start_date=normalized_start,
                end_date=normalized_end,
                output_dir=image_cluster_dir,
                feature_method=image_feature_method,
                n_clusters=image_clusters,
            )
        image_clusters_by_vehicle[str(vehicle_id)] = cluster_result

    argument_correlation = calculate_statistical_argument_correlation(
        vehicle_ids=vehicles,
        start_date=normalized_start,
        end_date=normalized_end,
        weights=_normalize_weights(argument_weights or DEFAULT_ARGUMENT_WEIGHTS),
        image_cluster_results=image_clusters_by_vehicle,
        route_cluster_output_dir=run_dir / "argument_route_clusters",
        route_feature_method=image_feature_method,
        route_clusters=image_clusters,
    )
    _write_json(run_dir / "argument_correlations.json", argument_correlation)

    argument_pairs = _target_pair_rows(
        argument_correlation["pairwise_argument_correlations"],
        target_vehicle,
    )
    truck_appearance = _truck_appearance_frequencies(argument_pairs, other_vehicles)

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
    _write_json(run_dir / "appearance_frequencies.json", {
        "truck_appearance_frequencies": {
            str(vehicle_id): values
            for vehicle_id, values in truck_appearance.items()
        },
        "target_route_appearance_frequencies": target_route_appearances,
        "route_appearance_frequencies": route_appearances,
    })
    _write_json(run_dir / "final_two_day_scenario.json", final_scenario)

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
        },
        "method": {
            "step_1": "Represent each truck-day with stop-zone, GPS image, route distance, and fleet metadata.",
            "step_2": "Build each truck's period stop-profile vector and compute stop-profile correlations.",
            "step_3": "Use the statistical argument-correlation method to combine stop-profile correlation, representative route-image similarity, and fleet metadata similarities.",
            "step_4": "Cluster each other truck's route images and calculate route appearance by cluster-day count divided by selected-period length.",
            "step_5": "Cluster the target truck's route images and sample the target route by its own route frequency.",
            "step_6": "Sample the final n-day scenario with the target truck route plus one other-truck route using joint scenario frequencies.",
            "truck_appearance_frequency": "Positive target-to-other argument correlations normalized across other trucks.",
            "route_appearance_frequency": "Cluster days divided by selected period length.",
            "joint_scenario_frequency": "Truck appearance frequency multiplied by route appearance frequency.",
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
            "summary": str(run_dir / "scenario_summary.json"),
            "data_representations": str(run_dir / "data_representations.json"),
            "stop_profile_correlations": str(run_dir / "stop_profile_correlations.json"),
            "argument_correlations": str(run_dir / "argument_correlations.json"),
            "image_clusters": str(run_dir / "image_clusters.json"),
            "appearance_frequencies": str(run_dir / "appearance_frequencies.json"),
            "final_scenario": str(run_dir / "final_two_day_scenario.json"),
        },
    }
    _write_json(run_dir / "scenario_summary.json", result)
    return result


def _route_distance_km(lookup: TruckDayLookup, vehicle_id: int, date: str) -> Optional[float]:
    representation = lookup.get(vehicle_id, date)
    if representation.route_distance is None:
        return None
    return representation.route_distance.total_distance_km


def _relative_error(expected: Optional[float], actual: Optional[float]) -> Optional[float]:
    if expected is None or actual is None:
        return None
    denominator = max(abs(actual), 1e-8)
    if denominator == 1e-8 and abs(expected) < 1e-8:
        return 0.0
    return abs(expected - actual) / denominator


def _validate_scenario_distances(
    scenario_result: dict,
    validation_dates: list[str],
    vehicles: list[int],
) -> dict:
    lookup = TruckDayLookup()
    generated_distances: dict[int, list[float]] = {vehicle_id: [] for vehicle_id in vehicles}
    real_distances: dict[int, list[float]] = {vehicle_id: [] for vehicle_id in vehicles}

    for day_index, day in enumerate(scenario_result["final_scenario"]["days"]):
        real_date = validation_dates[day_index % len(validation_dates)]
        participant_by_vehicle = {
            int(participant["vehicle_id"]): participant
            for participant in (day["target_vehicle"], day["other_vehicle"])
        }
        for vehicle_id in vehicles:
            participant = participant_by_vehicle.get(vehicle_id)
            generated_distance = (
                _route_distance_km(lookup, vehicle_id, participant["sampled_route_date"])
                if participant is not None
                else 0.0
            )
            real_distance = _route_distance_km(lookup, vehicle_id, real_date)
            generated_distances[vehicle_id].append(generated_distance)
            if real_distance is not None:
                real_distances[vehicle_id].append(real_distance)

    truck_rows = []
    generated_fleet_total = 0.0
    real_fleet_total = 0.0
    valid_fleet_count = 0
    for vehicle_id in vehicles:
        generated_values = generated_distances.get(vehicle_id, [])
        real_values = real_distances.get(vehicle_id, [])
        generated_average = sum(generated_values) / len(generated_values) if generated_values else None
        real_average = sum(real_values) / len(real_values) if real_values else None
        error = _relative_error(generated_average, real_average)
        if generated_average is not None and real_average is not None:
            generated_fleet_total += generated_average
            real_fleet_total += real_average
            valid_fleet_count += 1
        truck_rows.append(
            {
                "vehicle_id": vehicle_id,
                "generated_average_distance_km": generated_average,
                "real_average_distance_km": real_average,
                "relative_distance_error": error,
                "generated_distance_count": len(generated_values),
                "real_distance_count": len(real_values),
            }
        )

    valid_truck_errors = [
        row["relative_distance_error"]
        for row in truck_rows
        if row["relative_distance_error"] is not None
    ]
    mean_individual_error = (
        sum(valid_truck_errors) / len(valid_truck_errors)
        if valid_truck_errors
        else None
    )
    fleet_total_error = (
        _relative_error(generated_fleet_total, real_fleet_total)
        if valid_fleet_count
        else None
    )
    return {
        "validation_dates": validation_dates,
        "goal": "Compare generated sampled-route distances against random real dates in the selected period.",
        "fleet_scope": "All requested vehicles are validated together; non-sampled other trucks contribute generated distance 0 for that scenario day.",
        "metric": "mean_individual_relative_distance_error",
        "combined_error": mean_individual_error,
        "mean_individual_relative_distance_error": mean_individual_error,
        "fleet_total_relative_distance_error": fleet_total_error,
        "generated_fleet_average_distance_km": generated_fleet_total if valid_fleet_count else None,
        "real_fleet_average_distance_km": real_fleet_total if valid_fleet_count else None,
        "valid_fleet_count": valid_fleet_count,
        "trucks": truck_rows,
    }


def _neighbor_weights(
    weights: dict[str, float],
    step: float = 0.15,
    candidate_count: int = 3,
) -> list[dict[str, float]]:
    candidates = [_normalize_weights(weights)]
    for key in DEFAULT_ARGUMENT_WEIGHTS:
        adjusted = dict(weights)
        adjusted[key] = adjusted.get(key, 0.0) + step
        candidates.append(_normalize_weights(adjusted))
    return candidates[: max(1, candidate_count)]


def _error_value(entry: dict) -> float:
    error = entry["validation"]["mean_individual_relative_distance_error"]
    return error if error is not None else float("inf")


def _average(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _aggregate_seed_validations(seed_entries: list[dict]) -> dict:
    individual_errors = [
        entry["validation"]["mean_individual_relative_distance_error"]
        for entry in seed_entries
        if entry["validation"]["mean_individual_relative_distance_error"] is not None
    ]
    fleet_errors = [
        entry["validation"]["fleet_total_relative_distance_error"]
        for entry in seed_entries
        if entry["validation"]["fleet_total_relative_distance_error"] is not None
    ]
    return {
        "metric": "average_mean_individual_relative_distance_error_across_scenario_seeds",
        "mean_individual_relative_distance_error": _average(individual_errors),
        "fleet_total_relative_distance_error": _average(fleet_errors),
        "seed_count": len(seed_entries),
        "valid_individual_error_count": len(individual_errors),
        "valid_fleet_error_count": len(fleet_errors),
        "seed_results": seed_entries,
    }


def build_validation_loop(
    target_vehicle: int,
    other_vehicles: list[int],
    start_date: str,
    end_date: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    image_feature_method: str = "pca",
    image_clusters: int = 10,
    scenario_days: int = 2,
    seed: int = 0,
    validation_seed: int = 100,
    threshold: float = 0.25,
    max_iterations: int = 5,
    validation_scenario_seed_count: int = 5,
    validation_candidate_count: int = 3,
) -> dict:
    normalized_start = _normalize_date(start_date)
    normalized_end = _normalize_date(end_date)
    period_dates = _date_range(normalized_start, normalized_end)
    rng = random.Random(validation_seed)
    validation_dates = [rng.choice(period_dates) for _ in range(scenario_days)]

    run_label = (
        f"target_{target_vehicle}_others_{'_'.join(str(vehicle_id) for vehicle_id in other_vehicles)}_"
        f"{normalized_start}_{normalized_end}_validation_loop"
    )
    loop_dir = output_dir / run_label
    loop_dir.mkdir(parents=True, exist_ok=True)
    vehicles = [target_vehicle, *other_vehicles]
    image_cluster_cache = _build_image_cluster_cache(
        vehicle_ids=vehicles,
        start_date=normalized_start,
        end_date=normalized_end,
        output_dir=loop_dir / "cluster_cache",
        image_feature_method=image_feature_method,
        image_clusters=image_clusters,
    )
    _write_json(loop_dir / "cached_image_clusters.json", image_cluster_cache)

    current_weights = dict(DEFAULT_ARGUMENT_WEIGHTS)
    scenario_seeds = [seed + offset for offset in range(validation_scenario_seed_count)]
    history = []
    best_entry = None
    for iteration in range(1, max_iterations + 1):
        iteration_candidates = []
        for candidate_index, candidate_weights in enumerate(
            _neighbor_weights(current_weights, candidate_count=validation_candidate_count)
        ):
            candidate_dir = loop_dir / f"iteration_{iteration}" / f"candidate_{candidate_index}"
            seed_entries = []
            for scenario_seed in scenario_seeds:
                seed_dir = candidate_dir / f"seed_{scenario_seed}"
                scenario_result = build_scenario_generation(
                    target_vehicle=target_vehicle,
                    other_vehicles=other_vehicles,
                    start_date=normalized_start,
                    end_date=normalized_end,
                    output_dir=output_dir,
                    image_feature_method=image_feature_method,
                    image_clusters=image_clusters,
                    scenario_days=scenario_days,
                    seed=scenario_seed,
                    run_dir=seed_dir,
                    argument_weights=candidate_weights,
                    precomputed_image_clusters=image_cluster_cache,
                )
                seed_validation = _validate_scenario_distances(
                    scenario_result,
                    validation_dates,
                    vehicles,
                )
                seed_entry = {
                    "seed": scenario_seed,
                    "scenario_summary": scenario_result["output_files"]["summary"],
                    "final_scenario": scenario_result["output_files"]["final_scenario"],
                    "validation": seed_validation,
                }
                _write_json(seed_dir / "validation.json", seed_validation)
                seed_entries.append(seed_entry)

            seed_entries.sort(
                key=lambda entry: (
                    entry["validation"]["mean_individual_relative_distance_error"]
                    if entry["validation"]["mean_individual_relative_distance_error"] is not None
                    else float("inf")
                )
            )
            validation = _aggregate_seed_validations(seed_entries)
            entry = {
                "iteration": iteration,
                "candidate": candidate_index,
                "weights": candidate_weights,
                "scenario_seeds": scenario_seeds,
                "scenario_summary": seed_entries[0]["scenario_summary"] if seed_entries else None,
                "final_scenario": seed_entries[0]["final_scenario"] if seed_entries else None,
                "best_seed": seed_entries[0]["seed"] if seed_entries else None,
                "validation": validation,
            }
            _write_json(candidate_dir / "candidate_validation_summary.json", validation)
            iteration_candidates.append(entry)
            if best_entry is None or _error_value(entry) < _error_value(best_entry):
                best_entry = entry

        iteration_candidates.sort(key=_error_value)
        best_iteration_entry = iteration_candidates[0]
        history.append(
            {
                "iteration": iteration,
                "selected_candidate": best_iteration_entry["candidate"],
                "selected_weights": best_iteration_entry["weights"],
                "selected_validation_error": best_iteration_entry["validation"]["mean_individual_relative_distance_error"],
                "candidates": iteration_candidates,
            }
        )
        current_weights = best_iteration_entry["weights"]
        if (
            best_iteration_entry["validation"]["mean_individual_relative_distance_error"] is not None
            and best_iteration_entry["validation"]["mean_individual_relative_distance_error"] <= threshold
        ):
            break

    result = {
        "inputs": {
            "target_vehicle": target_vehicle,
            "other_vehicles": other_vehicles,
            "start_date": normalized_start,
            "end_date": normalized_end,
            "scenario_days": scenario_days,
            "seed": seed,
            "validation_scenario_seeds": scenario_seeds,
            "validation_seed": validation_seed,
            "threshold": threshold,
            "max_iterations": max_iterations,
            "validation_scenario_seed_count": validation_scenario_seed_count,
            "validation_candidate_count": validation_candidate_count,
            "validation_dates": validation_dates,
            "cluster_cache_file": str(loop_dir / "cached_image_clusters.json"),
        },
        "status": (
            "threshold_reached"
            if best_entry
            and best_entry["validation"]["mean_individual_relative_distance_error"] is not None
            and best_entry["validation"]["mean_individual_relative_distance_error"] <= threshold
            else "max_iterations_reached"
        ),
        "best": best_entry,
        "history": history,
    }
    _write_json(loop_dir / "validation_loop_summary.json", result)
    return result


def build_multi_seed_scenarios(
    target_vehicle: int,
    other_vehicles: list[int],
    start_date: str,
    end_date: str,
    seeds: list[int],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    image_feature_method: str = "pca",
    image_clusters: int = 10,
    scenario_days: int = 2,
) -> dict:
    normalized_start = _normalize_date(start_date)
    normalized_end = _normalize_date(end_date)
    run_label = (
        f"target_{target_vehicle}_others_{'_'.join(str(vehicle_id) for vehicle_id in other_vehicles)}_"
        f"{normalized_start}_{normalized_end}_multi_seed"
    )
    multi_seed_dir = output_dir / run_label
    multi_seed_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for seed in seeds:
        seed_dir = multi_seed_dir / f"seed_{seed}"
        result = build_scenario_generation(
            target_vehicle=target_vehicle,
            other_vehicles=other_vehicles,
            start_date=normalized_start,
            end_date=normalized_end,
            output_dir=output_dir,
            image_feature_method=image_feature_method,
            image_clusters=image_clusters,
            scenario_days=scenario_days,
            seed=seed,
            run_dir=seed_dir,
        )
        runs.append(
            {
                "seed": seed,
                "summary": result["output_files"]["summary"],
                "final_scenario": result["output_files"]["final_scenario"],
                "sampled_days": result["final_scenario"]["days"],
            }
        )

    index = {
        "inputs": {
            "target_vehicle": target_vehicle,
            "other_vehicles": other_vehicles,
            "start_date": normalized_start,
            "end_date": normalized_end,
            "scenario_days": scenario_days,
            "seeds": seeds,
            "image_feature_method": image_feature_method,
            "image_clusters": image_clusters,
        },
        "multi_seed_dir": str(multi_seed_dir),
        "runs": runs,
    }
    _write_json(multi_seed_dir / "multi_seed_index.json", index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Govern the full scenario generation pipeline from truck-day data "
            "representation, correlations, argument correlations, and image clusters."
        )
    )
    parser.add_argument("--target-vehicle", type=int, required=True)
    parser.add_argument("--other-vehicles", nargs="+", type=int, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-feature-method", choices=("pca", "autoencoder"), default="pca")
    parser.add_argument("--image-clusters", type=int, default=10)
    parser.add_argument("--scenario-days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-loop", action="store_true")
    parser.add_argument("--validation-threshold", type=float, default=0.25)
    parser.add_argument("--validation-seed", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument(
        "--validation-scenario-seed-count",
        type=int,
        default=5,
        help="Number of scenario seeds to evaluate per weight candidate in the validation loop.",
    )
    parser.add_argument(
        "--validation-candidate-count",
        type=int,
        default=3,
        help="Number of weight candidates to evaluate per validation-loop iteration.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Run multiple seeds into a separate multi-seed folder.",
    )
    args = parser.parse_args()

    if args.validation_loop:
        result = build_validation_loop(
            target_vehicle=args.target_vehicle,
            other_vehicles=args.other_vehicles,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            image_feature_method=args.image_feature_method,
            image_clusters=args.image_clusters,
            scenario_days=args.scenario_days,
            seed=args.seed,
            validation_seed=args.validation_seed,
            threshold=args.validation_threshold,
            max_iterations=args.max_iterations,
            validation_scenario_seed_count=args.validation_scenario_seed_count,
            validation_candidate_count=args.validation_candidate_count,
        )
        best = result["best"]
        best_error = best["validation"]["mean_individual_relative_distance_error"] if best else None
        best_error_text = "n/a" if best_error is None else f"{best_error:.3f}"
        print(f"Validation loop status: {result['status']}")
        print(f"Best validation error: {best_error_text}")
        print(
            "Best final scenario: "
            + (best["final_scenario"] if best else "n/a")
        )
        print(
            "Validation loop summary: "
            + str(
                args.output_dir
                / (
                    f"target_{args.target_vehicle}_others_{'_'.join(str(vehicle_id) for vehicle_id in args.other_vehicles)}_"
                    f"{_normalize_date(args.start_date)}_{_normalize_date(args.end_date)}_validation_loop"
                )
                / "validation_loop_summary.json"
            )
        )
        return

    if args.seeds:
        result = build_multi_seed_scenarios(
            target_vehicle=args.target_vehicle,
            other_vehicles=args.other_vehicles,
            start_date=args.start_date,
            end_date=args.end_date,
            seeds=args.seeds,
            output_dir=args.output_dir,
            image_feature_method=args.image_feature_method,
            image_clusters=args.image_clusters,
            scenario_days=args.scenario_days,
        )
        print(f"Wrote multi-seed index to {result['multi_seed_dir']}/multi_seed_index.json")
        for run in result["runs"]:
            print(f"  Seed {run['seed']}: {run['final_scenario']}")
        return

    result = build_scenario_generation(
        target_vehicle=args.target_vehicle,
        other_vehicles=args.other_vehicles,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        image_feature_method=args.image_feature_method,
        image_clusters=args.image_clusters,
        scenario_days=args.scenario_days,
        seed=args.seed,
    )

    print(f"Wrote scenario summary to {result['output_files']['summary']}")
    print("Truck appearance frequencies:")
    for vehicle_id, values in result["truck_appearance_frequencies"].items():
        print(
            f"  Vehicle {vehicle_id}: "
            f"correlation={values['raw_total_correlation']:.3f}, "
            f"appearance_frequency={values['appearance_frequency']:.3f}"
        )
    print("Top route appearances:")
    for row in result["route_appearance_frequencies"][:10]:
        print(
            f"  Vehicle {row['vehicle_id']} cluster {row['cluster']}: "
            f"route_frequency={row['route_appearance_frequency']:.3f}, "
            f"joint_frequency={row['joint_scenario_frequency']:.3f}"
        )
    print(f"Final scenario written to {result['output_files']['final_scenario']}")
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
