import argparse
import csv
import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from multi_truck_scenario import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_PASSES_FILE,
    _fit_kmeans,
    _fit_pca,
    _image_similarity,
    _load_image_features,
    _load_vehicle_passes,
    _pass_correlation,
)


DEFAULT_OUTPUT_ROOT = Path("closed_loop_runs")
DEFAULT_ROUTE_PATTERN_CLUSTERS = 10
DEFAULT_BEHAVIOR_CLUSTERS = 3
DEFAULT_STOP_PROFILE_WEIGHT = 0.8

DEFAULT_WEIGHTS = {
    "pass_score": 0.25,
    "image_similarity": 0.25,
    "distance_similarity": 0.3,
    "purpose_match": 0.1,
    "temperature_similarity": 0.1,
}

WEIGHT_KEYS = tuple(DEFAULT_WEIGHTS)


def _date_in_window(date: str, start_date: str, end_date: str) -> bool:
    return start_date <= date <= end_date


def _filter_by_date(data: dict[str, object], start_date: str, end_date: str) -> dict[str, object]:
    return {
        date: value
        for date, value in data.items()
        if _date_in_window(date, start_date, end_date)
    }


def _load_temperature(path: Optional[Path], target_year: Optional[int] = None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(line for line in file if not line.startswith("#"))
        if reader.fieldnames is None:
            return {}

        date_column = None
        for candidate in ("date", "local_time", "time"):
            if candidate in reader.fieldnames:
                date_column = candidate
                break

        if date_column is None:
            raise ValueError(f"{path} needs a date, local_time, or time column.")

        temp_column = None
        for candidate in ("temperature", "temp", "temperature_f", "temperature_c", "t2m"):
            if candidate in reader.fieldnames:
                temp_column = candidate
                break

        if temp_column is None:
            raise ValueError(f"{path} needs a temperature column.")

        daily_temperatures: dict[str, list[float]] = {}
        for row in reader:
            raw_date = row.get(date_column)
            raw_temperature = row.get(temp_column)
            if not raw_date or raw_temperature in ("", None):
                continue

            date = raw_date[:10]
            if target_year is not None:
                date = f"{target_year}{date[4:]}"
            daily_temperatures.setdefault(date, []).append(float(raw_temperature))

        return {
            date: sum(values) / len(values)
            for date, values in daily_temperatures.items()
        }


def _load_metadata(path: Optional[Path]) -> dict[int, dict[str, str]]:
    if path is None or not path.exists():
        return {}

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return {}

        vehicle_column = "VehicleId" if "VehicleId" in reader.fieldnames else "vehicle_id"
        return {
            int(row[vehicle_column]): row
            for row in reader
            if row.get(vehicle_column)
        }


def _load_feedback(path: Optional[Path]) -> dict[tuple[int, int], float]:
    if path is None or not path.exists():
        return {}

    feedback = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return {}

        for row in reader:
            left = int(row.get("vehicle_a") or row.get("VehicleA"))
            right = int(row.get("vehicle_b") or row.get("VehicleB"))
            if "feedback_score" in row and row["feedback_score"] != "":
                label = float(row["feedback_score"])
            elif "label" in row:
                label = 1.0 if row["label"].lower() in ("1", "true", "related", "shared_route") else 0.0
            else:
                continue
            feedback[tuple(sorted((left, right)))] = max(0.0, min(1.0, label))
    return feedback


def _load_distances(path: Optional[Path]) -> dict[int, dict[str, float]]:
    if path is None or not path.exists():
        return {}

    distances: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return {}

        vehicle_column = "VehicleId" if "VehicleId" in reader.fieldnames else "vehicle_id"
        date_column = "date" if "date" in reader.fieldnames else "Date"

        distance_column = None
        for candidate in (
            "total_distance_km",
            "total_distance_miles",
            "Daily_distance",
            "daily_distance",
            "distance",
        ):
            if candidate in reader.fieldnames:
                distance_column = candidate
                break

        if distance_column is None:
            raise ValueError(f"{path} needs a distance column.")

        for row in reader:
            if not row.get(vehicle_column) or not row.get(date_column):
                continue
            raw_distance = row.get(distance_column)
            if raw_distance in ("", None):
                continue
            vehicle_id = int(row[vehicle_column])
            date = row[date_column][:10]
            distance = float(raw_distance)
            if distance_column == "total_distance_miles":
                distance *= 1.609344
            distances.setdefault(vehicle_id, {})[date] = distance

    return distances


def _average_temperature_similarity(
    left_dates: set[str],
    right_dates: set[str],
    temperatures: dict[str, float],
) -> tuple[Optional[float], Optional[float], int]:
    if not temperatures:
        return None, None, 0

    left_values = [temperatures[date] for date in left_dates if date in temperatures]
    right_values = [temperatures[date] for date in right_dates if date in temperatures]
    if not left_values or not right_values:
        return None, None, 0

    left_avg = sum(left_values) / len(left_values)
    right_avg = sum(right_values) / len(right_values)
    similarity = 1.0 - min(abs(left_avg - right_avg) / 50.0, 1.0)
    return similarity, (left_avg + right_avg) / 2.0, min(len(left_values), len(right_values))


def _purpose_match(left_id: int, right_id: int, metadata: dict[int, dict[str, str]]) -> Optional[float]:
    if not metadata:
        return None

    left = metadata.get(left_id, {})
    right = metadata.get(right_id, {})
    purpose_left = left.get("purpose") or left.get("truck_purpose")
    purpose_right = right.get("purpose") or right.get("truck_purpose")
    if not purpose_left or not purpose_right:
        return None
    return 1.0 if purpose_left == purpose_right else 0.0


def _vehicle_purpose(vehicle_id: int, metadata: dict[int, dict[str, str]]) -> str:
    row = metadata.get(vehicle_id, {})
    return row.get("purpose") or row.get("truck_purpose") or "unknown"


def _cosine_similarity_values(left: list[float], right: list[float]) -> Optional[float]:
    if len(left) != len(right) or not left:
        return None
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    denominator = left_norm * right_norm
    if denominator == 0:
        return None
    return numerator / denominator


def _build_truck_stop_behavior_graph(
    vehicle_ids: list[int],
    passes_by_vehicle: dict[int, dict[str, tuple[int, ...]]],
    zone_columns: list[str],
    metadata: dict[int, dict[str, str]],
    start_date: str,
    end_date: str,
    stop_profile_weight: float,
    n_clusters: int,
) -> dict:
    stop_profile_weight = max(0.0, min(1.0, stop_profile_weight))
    purpose_weight = 1.0 - stop_profile_weight

    stop_profiles: dict[int, list[float]] = {}
    edge_rows = []
    for vehicle_id in vehicle_ids:
        daily_passes = _filter_by_date(passes_by_vehicle.get(vehicle_id, {}), start_date, end_date)
        totals = [0.0] * len(zone_columns)
        for values in daily_passes.values():
            for index, value in enumerate(values):
                totals[index] += float(value)
        denominator = max(1, len(daily_passes))
        profile = [value / denominator for value in totals]
        stop_profiles[vehicle_id] = profile

        for zone, weight in zip(zone_columns, profile):
            edge_rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "stop": zone,
                    "weight": weight,
                    "frequency_days": totals[zone_columns.index(zone)],
                    "observed_days": len(daily_passes),
                }
            )

    purposes = sorted({_vehicle_purpose(vehicle_id, metadata) for vehicle_id in vehicle_ids})
    purpose_index = {purpose: index for index, purpose in enumerate(purposes)}
    purpose_vectors = {}
    for vehicle_id in vehicle_ids:
        vector = [0.0] * len(purposes)
        vector[purpose_index[_vehicle_purpose(vehicle_id, metadata)]] = 1.0
        purpose_vectors[vehicle_id] = vector

    similarity_rows = []
    similarity_matrix = []
    combined_vectors = []
    for left_id in vehicle_ids:
        row = []
        combined_vectors.append(
            [
                math.sqrt(stop_profile_weight) * value
                for value in stop_profiles[left_id]
            ]
            + [
                math.sqrt(purpose_weight) * value
                for value in purpose_vectors[left_id]
            ]
        )
        for right_id in vehicle_ids:
            stop_sim = _cosine_similarity_values(stop_profiles[left_id], stop_profiles[right_id])
            purpose_sim = 1.0 if _vehicle_purpose(left_id, metadata) == _vehicle_purpose(right_id, metadata) else 0.0
            if stop_sim is None:
                stop_sim = 1.0 if left_id == right_id else 0.0
            similarity = stop_profile_weight * stop_sim + purpose_weight * purpose_sim
            row.append(similarity)
            if left_id < right_id:
                similarity_rows.append(
                    {
                        "vehicle_a": left_id,
                        "vehicle_b": right_id,
                        "stop_profile_similarity": stop_sim,
                        "purpose_similarity": purpose_sim,
                        "rho": similarity,
                    }
                )
        similarity_matrix.append(row)

    if vehicle_ids:
        cluster_count = min(max(1, n_clusters), len(vehicle_ids))
        vectors = np.asarray(combined_vectors, dtype=float)
        if cluster_count == 1:
            labels = np.zeros(len(vehicle_ids), dtype=int)
        else:
            labels, _ = _fit_kmeans(vectors, cluster_count)
    else:
        labels = np.asarray([], dtype=int)

    cluster_summaries = []
    for cluster_id in sorted({int(label) for label in labels}):
        members = [
            vehicle_id
            for vehicle_id, label in zip(vehicle_ids, labels)
            if int(label) == cluster_id
        ]
        if not members:
            continue
        mean_profile = [
            sum(stop_profiles[vehicle_id][index] for vehicle_id in members) / len(members)
            for index in range(len(zone_columns))
        ]
        top_stops = [
            {"stop": zone, "mean_weight": weight}
            for zone, weight in sorted(zip(zone_columns, mean_profile), key=lambda item: item[1], reverse=True)[:3]
        ]
        purpose_counts: dict[str, int] = {}
        for vehicle_id in members:
            purpose = _vehicle_purpose(vehicle_id, metadata)
            purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1
        cluster_summaries.append(
            {
                "cluster": cluster_id,
                "vehicles": members,
                "size": len(members),
                "top_stops": top_stops,
                "purpose_counts": purpose_counts,
            }
        )

    return {
        "method": {
            "edge_weight": "w_is = alpha * normalized_stop_frequency; duration is omitted.",
            "truck_similarity": "rho_ij = lambda * sim(stop_profile_i, stop_profile_j) + (1-lambda) * sim(purpose_i, purpose_j).",
            "stop_profile_weight_lambda": stop_profile_weight,
            "purpose_weight": purpose_weight,
        },
        "stops": zone_columns,
        "truck_stop_edges": edge_rows,
        "truck_stop_profiles": {
            str(vehicle_id): {
                "purpose": _vehicle_purpose(vehicle_id, metadata),
                "weights": {
                    zone: stop_profiles[vehicle_id][index]
                    for index, zone in enumerate(zone_columns)
                },
            }
            for vehicle_id in vehicle_ids
        },
        "truck_similarity_matrix": {
            "vehicles": vehicle_ids,
            "rho": similarity_matrix,
            "pairwise": sorted(similarity_rows, key=lambda row: row["rho"], reverse=True),
        },
        "behavioral_clusters": cluster_summaries,
    }


def _pearson_similarity(left: list[float], right: list[float]) -> Optional[float]:
    if len(left) < 2 or len(left) != len(right):
        return None

    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_denominator = sum((a - left_mean) ** 2 for a in left)
    right_denominator = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_denominator * right_denominator)
    if denominator == 0:
        return None
    correlation = numerator / denominator
    return max(0.0, min(1.0, (correlation + 1.0) / 2.0))


def _distance_similarity(
    left: dict[str, float],
    right: dict[str, float],
) -> tuple[Optional[float], int, Optional[float], Optional[float], Optional[float], Optional[float]]:
    common_dates = sorted(set(left) & set(right))
    if not common_dates:
        return None, 0, None, None, None, None

    left_values = [left[date] for date in common_dates]
    right_values = [right[date] for date in common_dates]
    left_total = sum(left_values)
    right_total = sum(right_values)
    max_total = max(left_total, right_total)
    total_match = 1.0 if max_total == 0 else 1.0 - min(abs(left_total - right_total) / max_total, 1.0)

    daily_matches = []
    for left_value, right_value in zip(left_values, right_values):
        denominator = max(left_value, right_value)
        if denominator == 0:
            daily_matches.append(1.0)
        else:
            daily_matches.append(1.0 - min(abs(left_value - right_value) / denominator, 1.0))
    daily_match = sum(daily_matches) / len(daily_matches)

    pattern_match = _pearson_similarity(left_values, right_values)
    pieces = [total_match, daily_match]
    if pattern_match is not None:
        pieces.append(pattern_match)

    return sum(pieces) / len(pieces), len(common_dates), left_total, right_total, daily_match, pattern_match


def _score_features(features: dict, weights: dict[str, float]) -> Optional[float]:
    weighted_values = []
    for key, weight in weights.items():
        value = features.get(key)
        if value is None:
            continue
        weighted_values.append((float(value), float(weight)))

    if not weighted_values:
        return None

    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight == 0:
        return None
    return sum(value * weight for value, weight in weighted_values) / total_weight


def _scenario_label(score: Optional[float]) -> str:
    if score is None:
        return "insufficient_data"
    if score >= 0.75:
        return "shared_route_behavior"
    if score >= 0.55:
        return "partially_shared_route_behavior"
    return "mostly_independent_route_behavior"


def _stability_label(delta: Optional[float], tolerance: float) -> str:
    if delta is None:
        return "insufficient_data"
    if abs(delta) <= tolerance:
        return "stable_close_to_training"
    if delta > 0:
        return "stronger_than_training"
    return "weaker_than_training"


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    normalized_input = {
        key: max(0.0, float(weights.get(key, 0.0)))
        for key in WEIGHT_KEYS
    }
    total = sum(normalized_input.values())
    if total == 0:
        return dict(DEFAULT_WEIGHTS)
    return {key: value / total for key, value in normalized_input.items()}


def _learn_weights(rows: list[dict], feedback: dict[tuple[int, int], float]) -> dict[str, float]:
    labeled_rows = [
        row
        for row in rows
        if tuple(sorted((row["vehicle_a"], row["vehicle_b"]))) in feedback
    ]
    if not labeled_rows:
        return dict(DEFAULT_WEIGHTS)

    feature_names = list(DEFAULT_WEIGHTS)
    weights = {name: 0.0 for name in feature_names}
    positive_count = 0.0

    for row in labeled_rows:
        label = feedback[tuple(sorted((row["vehicle_a"], row["vehicle_b"])))]
        positive_count += label
        for name in feature_names:
            value = row["features"].get(name)
            if value is not None:
                weights[name] += label * float(value)

    if positive_count == 0:
        return dict(DEFAULT_WEIGHTS)

    weights = {name: value / positive_count for name, value in weights.items()}
    return _normalize_weights(weights)


def _build_pair_features(
    vehicle_ids: list[int],
    image_features: dict[int, dict[str, np.ndarray]],
    passes_by_vehicle: dict[int, dict[str, tuple[int, ...]]],
    temperatures: dict[str, float],
    distances_by_vehicle: dict[int, dict[str, float]],
    metadata: dict[int, dict[str, str]],
    start_date: str,
    end_date: str,
    weights: dict[str, float],
) -> list[dict]:
    rows = []

    for left_id, right_id in combinations(vehicle_ids, 2):
        left_images = _filter_by_date(image_features[left_id], start_date, end_date)
        right_images = _filter_by_date(image_features[right_id], start_date, end_date)
        left_passes = _filter_by_date(passes_by_vehicle.get(left_id, {}), start_date, end_date)
        right_passes = _filter_by_date(passes_by_vehicle.get(right_id, {}), start_date, end_date)
        left_distances = _filter_by_date(distances_by_vehicle.get(left_id, {}), start_date, end_date)
        right_distances = _filter_by_date(distances_by_vehicle.get(right_id, {}), start_date, end_date)

        pass_corr, common_pass_dates, pass_features = _pass_correlation(left_passes, right_passes)
        pass_score = None if pass_corr is None else max(0.0, min(1.0, (pass_corr + 1.0) / 2.0))
        image_sim, common_image_dates, similar_dates = _image_similarity(left_images, right_images)
        distance_sim, common_distance_dates, left_total_distance, right_total_distance, daily_distance_match, distance_pattern_match = _distance_similarity(
            left_distances,
            right_distances,
        )
        temp_sim, average_temperature, temp_dates = _average_temperature_similarity(
            set(left_passes) | set(left_images),
            set(right_passes) | set(right_images),
            temperatures,
        )
        purpose = _purpose_match(left_id, right_id, metadata)

        features = {
            "pass_score": pass_score,
            "pass_correlation": pass_corr,
            "image_similarity": image_sim,
            "distance_similarity": distance_sim,
            "purpose_match": purpose,
            "temperature_similarity": temp_sim,
        }
        score = _score_features(features, weights)

        rows.append(
            {
                "vehicle_a": left_id,
                "vehicle_b": right_id,
                "features": features,
                "learned_similarity": score,
                "scenario_label": _scenario_label(score),
                "common_pass_dates": common_pass_dates,
                "pass_features_compared": pass_features,
                "common_image_dates": common_image_dates,
                "common_distance_dates": common_distance_dates,
                "left_total_distance_km": left_total_distance,
                "right_total_distance_km": right_total_distance,
                "daily_distance_match": daily_distance_match,
                "distance_pattern_match": distance_pattern_match,
                "temperature_dates": temp_dates,
                "average_temperature": average_temperature,
                "most_similar_image_dates": similar_dates,
            }
        )

    rows.sort(
        key=lambda row: row["learned_similarity"] if row["learned_similarity"] is not None else -1.0,
        reverse=True,
    )
    return rows


def _pair_key(row: dict) -> tuple[int, int]:
    return tuple(sorted((row["vehicle_a"], row["vehicle_b"])))


def _build_performance_vs_train(split_results: dict, tolerance: float = 0.05) -> dict:
    train_rows = {
        _pair_key(row): row
        for row in split_results["train"]["pairwise_relationships"]
    }
    performance = {}

    for split_name in ("validation", "test"):
        rows = []
        for row in split_results[split_name]["pairwise_relationships"]:
            train_row = train_rows.get(_pair_key(row))
            train_score = None if train_row is None else train_row["learned_similarity"]
            split_score = row["learned_similarity"]
            delta = None
            absolute_delta = None
            if train_score is not None and split_score is not None:
                delta = split_score - train_score
                absolute_delta = abs(delta)

            rows.append(
                {
                    "vehicle_a": row["vehicle_a"],
                    "vehicle_b": row["vehicle_b"],
                    "train_similarity": train_score,
                    f"{split_name}_similarity": split_score,
                    f"{split_name}_delta_from_train": delta,
                    f"{split_name}_absolute_delta_from_train": absolute_delta,
                    "stability_label": _stability_label(delta, tolerance),
                    "target": "smaller absolute delta means better closed-loop stability",
                }
            )

        rows.sort(
            key=lambda item: (
                item[f"{split_name}_absolute_delta_from_train"]
                if item[f"{split_name}_absolute_delta_from_train"] is not None
                else float("inf")
            )
        )
        valid_deltas = [
            item[f"{split_name}_absolute_delta_from_train"]
            for item in rows
            if item[f"{split_name}_absolute_delta_from_train"] is not None
        ]
        performance[split_name] = {
            "goal": "Keep calculated split correlation close to training correlation.",
            "tolerance": tolerance,
            "mean_absolute_delta_from_train": (
                sum(valid_deltas) / len(valid_deltas) if valid_deltas else None
            ),
            "pairwise": rows,
        }

    return performance


def _build_real_route_validation(split_results: dict) -> dict:
    validation = {}

    for split_name in ("validation", "test"):
        rows = []
        for row in split_results[split_name]["pairwise_relationships"]:
            scenario_score = row["learned_similarity"]
            distance_score = row["features"].get("distance_similarity")
            error = None
            if scenario_score is not None and distance_score is not None:
                error = abs(scenario_score - distance_score)

            rows.append(
                {
                    "vehicle_a": row["vehicle_a"],
                    "vehicle_b": row["vehicle_b"],
                    "scenario_similarity": scenario_score,
                    "real_distance_similarity": distance_score,
                    "absolute_error_vs_real_distance": error,
                    "common_distance_dates": row["common_distance_dates"],
                    "daily_distance_match": row["daily_distance_match"],
                    "distance_pattern_match": row["distance_pattern_match"],
                    "target": "smaller error means generated scenario is closer to real held-out route distance behavior",
                }
            )

        rows.sort(
            key=lambda item: (
                item["absolute_error_vs_real_distance"]
                if item["absolute_error_vs_real_distance"] is not None
                else float("inf")
            )
        )
        valid_errors = [
            item["absolute_error_vs_real_distance"]
            for item in rows
            if item["absolute_error_vs_real_distance"] is not None
        ]
        validation[split_name] = {
            "goal": "Compare generated scenario similarity against real held-out distance similarity.",
            "mean_absolute_error_vs_real_distance": (
                sum(valid_errors) / len(valid_errors) if valid_errors else None
            ),
            "pairwise": rows,
        }

    return validation


def _average_distance_in_window(
    distances: dict[str, float],
    start_date: str,
    end_date: str,
) -> tuple[Optional[float], int]:
    values = [
        distance
        for date, distance in distances.items()
        if _date_in_window(date, start_date, end_date)
    ]
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def _build_route_pattern_expectations(
    vehicle_ids: list[int],
    image_features: dict[int, dict[str, np.ndarray]],
    distances_by_vehicle: dict[int, dict[str, float]],
    train_start: str,
    train_end: str,
    n_clusters: int = DEFAULT_ROUTE_PATTERN_CLUSTERS,
) -> dict[int, dict]:
    expectations = {}

    for vehicle_id in vehicle_ids:
        train_dates = [
            date
            for date in sorted(image_features.get(vehicle_id, {}))
            if _date_in_window(date, train_start, train_end)
        ]
        if not train_dates:
            expectations[vehicle_id] = {
                "generated_average_distance_km": None,
                "selected_cluster": None,
                "route_patterns": [],
                "method": "no_training_images",
            }
            continue

        raw_features = np.vstack([image_features[vehicle_id][date] for date in train_dates])
        cluster_count = min(n_clusters, len(train_dates))
        if cluster_count < 2:
            labels = np.zeros(len(train_dates), dtype=int)
        else:
            n_components = min(21, len(train_dates) - 1, raw_features.shape[1])
            features = _fit_pca(raw_features, n_components)
            labels, _ = _fit_kmeans(features, cluster_count)

        route_patterns = []
        weighted_distance_sum = 0.0
        weighted_distance_weight = 0.0

        for cluster_id in sorted({int(label) for label in labels}):
            cluster_dates = [
                date
                for date, label in zip(train_dates, labels)
                if int(label) == cluster_id
            ]
            cluster_distances = [
                distances_by_vehicle.get(vehicle_id, {}).get(date)
                for date in cluster_dates
                if distances_by_vehicle.get(vehicle_id, {}).get(date) is not None
            ]
            prior = len(cluster_dates) / len(train_dates)
            average_distance = (
                sum(cluster_distances) / len(cluster_distances)
                if cluster_distances
                else None
            )
            if average_distance is not None:
                weighted_distance_sum += prior * average_distance
                weighted_distance_weight += prior

            route_patterns.append(
                {
                    "cluster": cluster_id,
                    "prior_probability": prior,
                    "train_image_days": len(cluster_dates),
                    "train_distance_days": len(cluster_distances),
                    "average_distance_km": average_distance,
                    "example_dates": cluster_dates[:5],
                }
            )

        route_patterns.sort(key=lambda row: row["prior_probability"], reverse=True)
        generated_distance = (
            weighted_distance_sum / weighted_distance_weight
            if weighted_distance_weight
            else None
        )

        expectations[vehicle_id] = {
            "generated_average_distance_km": generated_distance,
            "selected_cluster": route_patterns[0]["cluster"] if route_patterns else None,
            "selected_cluster_prior": route_patterns[0]["prior_probability"] if route_patterns else None,
            "route_patterns": route_patterns,
            "method": "training_route_pattern_prior_weighted_distance",
        }

    return expectations


def _relative_error(expected: Optional[float], actual: Optional[float]) -> Optional[float]:
    if expected is None or actual is None:
        return None
    denominator = max(abs(actual), 1e-8)
    if denominator == 1e-8 and abs(expected) < 1e-8:
        return 0.0
    return abs(expected - actual) / denominator


def _build_scenario_realism_validation(
    vehicle_ids: list[int],
    distances_by_vehicle: dict[int, dict[str, float]],
    splits: dict[str, tuple[str, str]],
    route_pattern_expectations: Optional[dict[int, dict]] = None,
    truck_weight: float = 0.7,
    fleet_weight: float = 0.3,
) -> dict:
    train_start, train_end = splits["train"]
    if route_pattern_expectations is None:
        route_pattern_expectations = {
            vehicle_id: {
                "generated_average_distance_km": _average_distance_in_window(
                    distances_by_vehicle.get(vehicle_id, {}),
                    train_start,
                    train_end,
                )[0],
                "route_patterns": [],
                "method": "training_average_distance",
            }
            for vehicle_id in vehicle_ids
        }

    validation = {
        "goal": (
            "Validate the generated multi-truck scenario against real held-out "
            "route distances, using each truck's training route-pattern priors as "
            "the generated expected route distance."
        ),
        "formula": (
            f"{truck_weight:.2f} * mean per-truck relative distance error + "
            f"{fleet_weight:.2f} * fleet-total relative distance error"
        ),
        "route_pattern_generation": "Each truck's route clusters are weighted by that truck's own training frequency prior.",
        "train_generated_route_expectations": [
            {"vehicle_id": vehicle_id, **route_pattern_expectations[vehicle_id]}
            for vehicle_id in vehicle_ids
        ],
    }

    for split_name in ("validation", "test"):
        start_date, end_date = splits[split_name]
        truck_rows = []
        generated_fleet_total = 0.0
        real_fleet_total = 0.0
        valid_fleet_count = 0

        for vehicle_id in vehicle_ids:
            expectation = route_pattern_expectations[vehicle_id]
            generated_distance = expectation["generated_average_distance_km"]
            real_distance, real_days = _average_distance_in_window(
                distances_by_vehicle.get(vehicle_id, {}),
                start_date,
                end_date,
            )
            error = _relative_error(generated_distance, real_distance)

            if generated_distance is not None and real_distance is not None:
                generated_fleet_total += generated_distance
                real_fleet_total += real_distance
                valid_fleet_count += 1

            truck_rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "generated_average_distance_km": generated_distance,
                    "real_average_distance_km": real_distance,
                    "relative_distance_error": error,
                    "selected_cluster": expectation.get("selected_cluster"),
                    "selected_cluster_prior": expectation.get("selected_cluster_prior"),
                    "generation_method": expectation.get("method"),
                    f"{split_name}_distance_days": real_days,
                    "target": "smaller error means this truck's generated route distance is closer to held-out ground truth",
                }
            )

        valid_truck_errors = [
            row["relative_distance_error"]
            for row in truck_rows
            if row["relative_distance_error"] is not None
        ]
        mean_truck_error = (
            sum(valid_truck_errors) / len(valid_truck_errors)
            if valid_truck_errors
            else None
        )
        fleet_total_error = (
            _relative_error(generated_fleet_total, real_fleet_total)
            if valid_fleet_count
            else None
        )
        scenario_error = None
        if mean_truck_error is not None and fleet_total_error is not None:
            scenario_error = truck_weight * mean_truck_error + fleet_weight * fleet_total_error

        validation[split_name] = {
            "start_date": start_date,
            "end_date": end_date,
            "scenario_realism_error": scenario_error,
            "mean_per_truck_relative_distance_error": mean_truck_error,
            "fleet_total_relative_distance_error": fleet_total_error,
            "generated_fleet_average_distance_km": generated_fleet_total if valid_fleet_count else None,
            "real_fleet_average_distance_km": real_fleet_total if valid_fleet_count else None,
            "valid_truck_count": valid_fleet_count,
            "trucks": truck_rows,
        }

    return validation


def _dominant_cluster_choices(route_pattern_expectations: dict[int, dict]) -> dict[int, dict]:
    choices = {}
    for vehicle_id, expectation in route_pattern_expectations.items():
        patterns = expectation.get("route_patterns", [])
        selected = patterns[0] if patterns else {}
        choices[int(vehicle_id)] = {
            "cluster": selected.get("cluster"),
            "expected_distance_km": selected.get("average_distance_km"),
            "cluster_prior": selected.get("prior_probability"),
            "method": "dominant_training_route_pattern",
            "rationale": "Highest-frequency training route pattern for this truck.",
        }
    return choices


def _build_recent_behavior_context(
    vehicle_ids: list[int],
    distances_by_vehicle: dict[int, dict[str, float]],
    route_pattern_expectations: dict[int, dict],
    context_start: Optional[str],
    context_end: Optional[str],
) -> Optional[dict]:
    if context_start is None or context_end is None:
        return None

    vehicles = {}
    for vehicle_id in vehicle_ids:
        recent_distance, recent_days = _average_distance_in_window(
            distances_by_vehicle.get(vehicle_id, {}),
            context_start,
            context_end,
        )
        cluster_matches = []
        for pattern in route_pattern_expectations.get(vehicle_id, {}).get("route_patterns", []):
            average_distance = pattern.get("average_distance_km")
            if average_distance is None or recent_distance is None:
                distance_gap = None
                relative_gap = None
            else:
                distance_gap = abs(average_distance - recent_distance)
                relative_gap = distance_gap / max(abs(recent_distance), 1e-8)
            cluster_matches.append(
                {
                    "cluster": pattern.get("cluster"),
                    "prior_probability": pattern.get("prior_probability"),
                    "average_distance_km": average_distance,
                    "recent_distance_gap_km": distance_gap,
                    "recent_relative_distance_gap": relative_gap,
                }
            )

        cluster_matches.sort(
            key=lambda row: (
                row["recent_relative_distance_gap"]
                if row["recent_relative_distance_gap"] is not None
                else float("inf")
            )
        )
        vehicles[str(vehicle_id)] = {
            "recent_start": context_start,
            "recent_end": context_end,
            "recent_average_distance_km": recent_distance,
            "recent_distance_days": recent_days,
            "closest_route_clusters_by_recent_distance": cluster_matches[:5],
        }

    return {
        "purpose": "Recent behavior context for target-period scenario generation.",
        "start_date": context_start,
        "end_date": context_end,
        "vehicles": vehicles,
    }


def _recent_context_cluster_choices(
    route_pattern_expectations: dict[int, dict],
    recent_behavior_context: Optional[dict],
) -> dict[int, dict]:
    if recent_behavior_context is None:
        return _dominant_cluster_choices(route_pattern_expectations)

    choices = {}
    vehicles = recent_behavior_context.get("vehicles", {})
    for vehicle_id, expectation in route_pattern_expectations.items():
        recent = vehicles.get(str(vehicle_id), {})
        matches = recent.get("closest_route_clusters_by_recent_distance", [])
        selected = matches[0] if matches else None
        if selected is None or selected.get("cluster") is None:
            choices[int(vehicle_id)] = _dominant_cluster_choices(route_pattern_expectations).get(int(vehicle_id), {})
            continue

        choices[int(vehicle_id)] = {
            "cluster": selected.get("cluster"),
            "expected_distance_km": selected.get("average_distance_km"),
            "cluster_prior": selected.get("prior_probability"),
            "method": "closest_to_recent_context_distance",
            "rationale": (
                f"Selected route cluster closest to recent average distance "
                f"from {recent_behavior_context['start_date']} to {recent_behavior_context['end_date']}."
            ),
        }

    return choices


def _build_validation_feedback_context(
    vehicle_ids: list[int],
    distances_by_vehicle: dict[int, dict[str, float]],
    route_pattern_expectations: dict[int, dict],
    validation_start: str,
    validation_end: str,
) -> dict:
    context = _build_recent_behavior_context(
        vehicle_ids,
        distances_by_vehicle,
        route_pattern_expectations,
        validation_start,
        validation_end,
    )
    if context is None:
        return {
            "purpose": "Validation feedback was requested but no validation context could be built.",
            "start_date": validation_start,
            "end_date": validation_end,
            "vehicles": {},
        }
    context["purpose"] = (
        "Validation feedback for revising route-cluster choices before generating the test scenario."
    )
    return context


def _build_cluster_choice_realism_validation(
    vehicle_ids: list[int],
    distances_by_vehicle: dict[int, dict[str, float]],
    splits: dict[str, tuple[str, str]],
    cluster_choices: dict[int, dict],
    truck_weight: float = 0.7,
    fleet_weight: float = 0.3,
) -> dict:
    validation = {
        "goal": "Validate selected route clusters against held-out real per-truck and fleet-total distances.",
        "formula": (
            f"{truck_weight:.2f} * mean per-truck relative distance error + "
            f"{fleet_weight:.2f} * fleet-total relative distance error"
        ),
        "cluster_choices": [
            {"vehicle_id": vehicle_id, **cluster_choices.get(vehicle_id, {})}
            for vehicle_id in vehicle_ids
        ],
    }

    for split_name in ("validation", "test"):
        start_date, end_date = splits[split_name]
        truck_rows = []
        generated_fleet_total = 0.0
        real_fleet_total = 0.0
        valid_fleet_count = 0

        for vehicle_id in vehicle_ids:
            choice = cluster_choices.get(vehicle_id, {})
            generated_distance = choice.get("expected_distance_km")
            real_distance, real_days = _average_distance_in_window(
                distances_by_vehicle.get(vehicle_id, {}),
                start_date,
                end_date,
            )
            error = _relative_error(generated_distance, real_distance)

            if generated_distance is not None and real_distance is not None:
                generated_fleet_total += generated_distance
                real_fleet_total += real_distance
                valid_fleet_count += 1

            truck_rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "selected_cluster": choice.get("cluster"),
                    "cluster_prior": choice.get("cluster_prior"),
                    "generated_distance_km": generated_distance,
                    "real_average_distance_km": real_distance,
                    "relative_distance_error": error,
                    f"{split_name}_distance_days": real_days,
                    "rationale": choice.get("rationale", ""),
                }
            )

        valid_truck_errors = [
            row["relative_distance_error"]
            for row in truck_rows
            if row["relative_distance_error"] is not None
        ]
        mean_truck_error = (
            sum(valid_truck_errors) / len(valid_truck_errors)
            if valid_truck_errors
            else None
        )
        fleet_total_error = (
            _relative_error(generated_fleet_total, real_fleet_total)
            if valid_fleet_count
            else None
        )
        scenario_error = None
        if mean_truck_error is not None and fleet_total_error is not None:
            scenario_error = truck_weight * mean_truck_error + fleet_weight * fleet_total_error

        validation[split_name] = {
            "start_date": start_date,
            "end_date": end_date,
            "scenario_realism_error": scenario_error,
            "mean_per_truck_relative_distance_error": mean_truck_error,
            "fleet_total_relative_distance_error": fleet_total_error,
            "generated_fleet_distance_km": generated_fleet_total if valid_fleet_count else None,
            "real_fleet_average_distance_km": real_fleet_total if valid_fleet_count else None,
            "valid_truck_count": valid_fleet_count,
            "trucks": truck_rows,
        }

    return validation


def _llm_json_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response did not contain a JSON object.")
    return json.loads(text[start : end + 1])


def _summarize_for_llm(run: dict) -> dict:
    return {
        "vehicles": run["vehicles"],
        "current_weights": run["learned_weights"],
        "performance_goal": run["performance_goal"],
        "feature_meaning": run["feature_meaning"],
        "truck_stop_behavior_graph": {
            "method": run["truck_stop_behavior_graph"]["method"],
            "behavioral_clusters": run["truck_stop_behavior_graph"]["behavioral_clusters"],
            "truck_similarity_matrix": run["truck_stop_behavior_graph"]["truck_similarity_matrix"],
        },
        "route_pattern_expectations": run["route_pattern_expectations"],
        "recent_behavior_context": run.get("recent_behavior_context"),
        "dominant_cluster_choices": run["dominant_cluster_choices"],
        "recent_context_cluster_choices": run.get("recent_context_cluster_choices"),
        "recent_context_cluster_choice_realism_validation": run.get("recent_context_cluster_choice_realism_validation"),
        "validation_feedback_context": run.get("validation_feedback_context"),
        "validation_feedback_cluster_choices": run.get("validation_feedback_cluster_choices"),
        "validation_feedback_cluster_choice_realism_validation": run.get("validation_feedback_cluster_choice_realism_validation"),
        "dominant_cluster_choice_realism_validation": run["dominant_cluster_choice_realism_validation"],
        "performance_vs_train": run["performance_vs_train"],
        "scenario_realism_validation": run["scenario_realism_validation"],
        "real_route_distance_validation": run["real_route_distance_validation"],
        "split_pair_features": {
            split_name: [
                {
                    "vehicle_a": row["vehicle_a"],
                    "vehicle_b": row["vehicle_b"],
                    "learned_similarity": row["learned_similarity"],
                    "features": row["features"],
                    "common_pass_dates": row["common_pass_dates"],
                    "common_image_dates": row["common_image_dates"],
                    "common_distance_dates": row["common_distance_dates"],
                    "daily_distance_match": row["daily_distance_match"],
                    "distance_pattern_match": row["distance_pattern_match"],
                    "scenario_label": row["scenario_label"],
                }
                for row in split["pairwise_relationships"]
            ]
            for split_name, split in run["splits"].items()
        },
    }


def _request_llm_weight_update(
    run: dict,
    model_name: Optional[str] = None,
    provider: str = "anthropic",
) -> dict:
    load_dotenv()
    load_dotenv("sample.env", override=False)
    load_dotenv("sample1.env", override=True)
    for env_key in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        if os.getenv(env_key):
            os.environ[env_key] = os.environ[env_key].strip()

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise RuntimeError("langchain-anthropic is required for --use-llm.") from e

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for --use-llm.")
        resolved_model = model_name or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        llm = ChatAnthropic(model=resolved_model)
    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise RuntimeError("langchain-openai is required for --llm-provider openai.") from e

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --llm-provider openai.")
        resolved_model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=resolved_model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    payload = _summarize_for_llm(run)
    prompt = f"""
You are updating feature weights for a closed-loop truck correlation system.

Goal:
- The generated multi-truck scenario should resemble the real held-out route behavior for the full set of trucks.
- Primary validation is scenario_realism_validation: per-truck route distance error plus fleet-total route distance error.
- Route patterns are truck-specific: common clusters should usually be more likely than rare exception clusters unless other evidence justifies changing them.
- Pairwise real_route_distance_validation is secondary evidence about whether truck relationships resemble held-out route behavior.
- Stability versus training is diagnostic only; do not optimize it at the expense of scenario realism.

Allowed weight keys:
{list(WEIGHT_KEYS)}

Rules:
- Return only one JSON object.
- Weight values must be non-negative and should sum to 1.0.
- If a feature is missing/null for most rows, reduce its weight.
- Preserve explainability; do not invent unavailable features.
- Include concise rationale and risks.

Input evidence:
{json.dumps(payload, indent=2)}

Return this schema:
{{
  "updated_weights": {{
    "pass_score": 0.0,
    "image_similarity": 0.0,
    "distance_similarity": 0.0,
    "purpose_match": 0.0,
    "temperature_similarity": 0.0
  }},
  "rationale": "short explanation",
  "expected_effect": "short explanation",
  "risks": ["short risk"]
}}
"""
    response = llm.invoke(prompt)
    parsed = _extract_json_object(_llm_json_text(response))
    updated_weights = _normalize_weights(parsed.get("updated_weights", {}))
    return {
        "provider": provider,
        "model": resolved_model,
        "updated_weights": updated_weights,
        "rationale": parsed.get("rationale", ""),
        "expected_effect": parsed.get("expected_effect", ""),
        "risks": parsed.get("risks", []),
    }


def _build_llm(
    model_name: Optional[str] = None,
    provider: str = "anthropic",
):
    load_dotenv()
    load_dotenv("sample.env", override=False)
    load_dotenv("sample1.env", override=True)
    for env_key in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        if os.getenv(env_key):
            os.environ[env_key] = os.environ[env_key].strip()

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise RuntimeError("langchain-anthropic is required for --use-llm.") from e

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for --use-llm.")
        resolved_model = model_name or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        return ChatAnthropic(model=resolved_model), resolved_model

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise RuntimeError("langchain-openai is required for --llm-provider openai.") from e

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --llm-provider openai.")
        resolved_model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=resolved_model), resolved_model

    raise ValueError(f"Unsupported LLM provider: {provider}")


def _cluster_lookup(route_pattern_expectations: dict[int, dict]) -> dict[int, dict[int, dict]]:
    lookup = {}
    for vehicle_id, expectation in route_pattern_expectations.items():
        lookup[int(vehicle_id)] = {
            int(pattern["cluster"]): pattern
            for pattern in expectation.get("route_patterns", [])
            if pattern.get("cluster") is not None
        }
    return lookup


def _request_llm_scenario_generation(
    run: dict,
    model_name: Optional[str] = None,
    provider: str = "anthropic",
) -> dict:
    llm, resolved_model = _build_llm(model_name, provider)
    payload = {
        "vehicles": run["vehicles"],
        "goal": (
            "Generate a multi-truck route scenario by choosing one route cluster per truck. "
            "Use only training evidence: truck-stop behavioral clusters, truck-specific route pattern priors, and train pairwise features."
        ),
        "truck_stop_behavior_graph": {
            "method": run["truck_stop_behavior_graph"]["method"],
            "behavioral_clusters": run["truck_stop_behavior_graph"]["behavioral_clusters"],
            "truck_similarity_matrix": run["truck_stop_behavior_graph"]["truck_similarity_matrix"],
        },
        "route_pattern_expectations": run["route_pattern_expectations"],
        "recent_behavior_context": run.get("recent_behavior_context"),
        "validation_feedback_context": run.get("validation_feedback_context"),
        "train_pair_features": [
            {
                "vehicle_a": row["vehicle_a"],
                "vehicle_b": row["vehicle_b"],
                "features": row["features"],
                "learned_similarity": row["learned_similarity"],
                "scenario_label": row["scenario_label"],
                "common_pass_dates": row["common_pass_dates"],
                "common_image_dates": row["common_image_dates"],
                "common_distance_dates": row["common_distance_dates"],
            }
            for row in run["splits"]["train"]["pairwise_relationships"]
        ],
    }
    prompt = f"""
You are a route-scenario generator for a fleet of trucks.

Task:
- Choose exactly one route cluster for each truck.
- Generate truck-to-truck correlation values directly, without using a fixed linear feature formula.
- Use the truck-stop behavioral clusters as the high-level scenario context.
- If recent_behavior_context is provided, generate the target scenario using that recent behavior as the strongest future-condition signal.
- If validation_feedback_context is provided, use it to revise route-cluster choices for the test scenario. This is allowed feedback; do not use test-period ground truth.
- Route clusters are truck-specific. A high-frequency cluster is usually more likely, but you may choose a lower-frequency cluster if distance, image, pass, or correlation evidence justifies it.
- Truck purpose is already represented in the truck similarity graph; parking duration is not used.
- Use only the training evidence in the input. Do not assume held-out validation/test results.

Rules:
- Return only one JSON object.
- Choose clusters only from the provided route_pattern_expectations for each truck.
- Correlations must be numbers from 0.0 to 1.0.
- Include concise rationales.

Input evidence:
{json.dumps(payload, indent=2)}

Return this schema:
{{
  "vehicle_cluster_choices": {{
    "689": {{"cluster": 0, "rationale": "short reason"}}
  }},
  "predicted_pairwise_correlations": [
    {{"vehicle_a": 689, "vehicle_b": 1994, "correlation": 0.0, "rationale": "short reason"}}
  ],
  "scenario_rationale": "short explanation",
  "risks": ["short risk"]
}}
"""
    response = llm.invoke(prompt)
    parsed = _extract_json_object(_llm_json_text(response))
    lookup = _cluster_lookup(run["route_pattern_expectations"])

    cluster_choices = {}
    raw_choices = parsed.get("vehicle_cluster_choices", {})
    for vehicle_id in run["vehicles"]:
        raw_choice = raw_choices.get(str(vehicle_id), raw_choices.get(vehicle_id, {}))
        selected_cluster = raw_choice.get("cluster")
        selected_pattern = None
        if selected_cluster is not None:
            selected_pattern = lookup.get(vehicle_id, {}).get(int(selected_cluster))
        if selected_pattern is None:
            fallback = _dominant_cluster_choices(run["route_pattern_expectations"]).get(vehicle_id, {})
            cluster_choices[vehicle_id] = {
                **fallback,
                "method": "llm_cluster_choice_fallback_to_dominant",
                "rationale": (
                    "LLM selected an unavailable cluster; used dominant training pattern. "
                    + str(raw_choice.get("rationale", ""))
                ).strip(),
            }
            continue

        cluster_choices[vehicle_id] = {
            "cluster": int(selected_pattern["cluster"]),
            "expected_distance_km": selected_pattern.get("average_distance_km"),
            "cluster_prior": selected_pattern.get("prior_probability"),
            "method": "llm_selected_route_pattern",
            "rationale": raw_choice.get("rationale", ""),
        }

    pairwise = []
    for row in parsed.get("predicted_pairwise_correlations", []):
        try:
            correlation = max(0.0, min(1.0, float(row["correlation"])))
            pairwise.append(
                {
                    "vehicle_a": int(row["vehicle_a"]),
                    "vehicle_b": int(row["vehicle_b"]),
                    "correlation": correlation,
                    "rationale": row.get("rationale", ""),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    return {
        "provider": provider,
        "model": resolved_model,
        "cluster_choices": cluster_choices,
        "predicted_pairwise_correlations": pairwise,
        "scenario_rationale": parsed.get("scenario_rationale", ""),
        "risks": parsed.get("risks", []),
    }


def _rescore_split_results(split_results: dict, weights: dict[str, float]) -> dict:
    rescored = json.loads(json.dumps(split_results))
    for split in rescored.values():
        for row in split["pairwise_relationships"]:
            row["learned_similarity"] = _score_features(row["features"], weights)
            row["scenario_label"] = _scenario_label(row["learned_similarity"])
        split["pairwise_relationships"].sort(
            key=lambda row: row["learned_similarity"] if row["learned_similarity"] is not None else -1.0,
            reverse=True,
        )
    return rescored


def _apply_llm_weight_update(
    run: dict,
    model_name: Optional[str] = None,
    provider: str = "anthropic",
) -> dict:
    update = _request_llm_weight_update(run, model_name, provider)
    llm_split_results = _rescore_split_results(run["splits"], update["updated_weights"])
    scenario_generation = _request_llm_scenario_generation(run, model_name, provider)
    run["llm_weight_update"] = update
    run["llm_rescored_splits"] = llm_split_results
    run["llm_performance_vs_train"] = _build_performance_vs_train(llm_split_results)
    run["llm_real_route_distance_validation"] = _build_real_route_validation(llm_split_results)
    run["llm_scenario_generation"] = scenario_generation
    run["llm_cluster_choice_realism_validation"] = _build_cluster_choice_realism_validation(
        run["vehicles"],
        run["_runtime_distances_by_vehicle"],
        {
            "train": (
                run["splits"]["train"]["start_date"],
                run["splits"]["train"]["end_date"],
            ),
            "validation": (
                run["splits"]["validation"]["start_date"],
                run["splits"]["validation"]["end_date"],
            ),
            "test": (
                run["splits"]["test"]["start_date"],
                run["splits"]["test"]["end_date"],
            ),
        },
        scenario_generation["cluster_choices"],
    )
    return run


def build_closed_loop_run(
    vehicle_ids: list[int],
    image_root: Path,
    passes_file: Path,
    temperature_file: Optional[Path],
    temperature_target_year: Optional[int],
    distance_file: Optional[Path],
    metadata_file: Optional[Path],
    feedback_file: Optional[Path],
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
    output_root: Path,
    route_pattern_clusters: int = DEFAULT_ROUTE_PATTERN_CLUSTERS,
    behavior_clusters: int = DEFAULT_BEHAVIOR_CLUSTERS,
    stop_profile_weight: float = DEFAULT_STOP_PROFILE_WEIGHT,
    scenario_context_start: Optional[str] = None,
    scenario_context_end: Optional[str] = None,
    use_validation_feedback: bool = False,
    use_llm: bool = False,
    llm_model: Optional[str] = None,
    llm_provider: str = "anthropic",
) -> dict:
    passes_by_vehicle, zone_columns = _load_vehicle_passes(passes_file)
    image_features = {
        vehicle_id: _load_image_features(image_root, vehicle_id)
        for vehicle_id in vehicle_ids
    }
    temperatures = _load_temperature(temperature_file, temperature_target_year)
    distances_by_vehicle = _load_distances(distance_file)
    metadata = _load_metadata(metadata_file)
    feedback = _load_feedback(feedback_file)
    truck_stop_behavior_graph = _build_truck_stop_behavior_graph(
        vehicle_ids,
        passes_by_vehicle,
        zone_columns,
        metadata,
        train_start,
        train_end,
        stop_profile_weight,
        behavior_clusters,
    )

    train_rows_for_learning = _build_pair_features(
        vehicle_ids,
        image_features,
        passes_by_vehicle,
        temperatures,
        distances_by_vehicle,
        metadata,
        train_start,
        train_end,
        DEFAULT_WEIGHTS,
    )
    learned_weights = _learn_weights(train_rows_for_learning, feedback)

    splits = {
        "train": (train_start, train_end),
        "validation": (validation_start, validation_end),
        "test": (test_start, test_end),
    }
    route_pattern_expectations = _build_route_pattern_expectations(
        vehicle_ids,
        image_features,
        distances_by_vehicle,
        train_start,
        train_end,
        route_pattern_clusters,
    )
    dominant_cluster_choices = _dominant_cluster_choices(route_pattern_expectations)
    recent_behavior_context = _build_recent_behavior_context(
        vehicle_ids,
        distances_by_vehicle,
        route_pattern_expectations,
        scenario_context_start,
        scenario_context_end,
    )
    recent_context_cluster_choices = _recent_context_cluster_choices(
        route_pattern_expectations,
        recent_behavior_context,
    )
    validation_feedback_context = None
    validation_feedback_cluster_choices = None
    if use_validation_feedback:
        validation_feedback_context = _build_validation_feedback_context(
            vehicle_ids,
            distances_by_vehicle,
            route_pattern_expectations,
            validation_start,
            validation_end,
        )
        validation_feedback_cluster_choices = _recent_context_cluster_choices(
            route_pattern_expectations,
            validation_feedback_context,
        )
    split_results = {}
    for split_name, (start_date, end_date) in splits.items():
        split_results[split_name] = {
            "start_date": start_date,
            "end_date": end_date,
            "pairwise_relationships": _build_pair_features(
                vehicle_ids,
                image_features,
                passes_by_vehicle,
                temperatures,
                distances_by_vehicle,
                metadata,
                start_date,
                end_date,
                learned_weights,
            ),
        }

    run = {
        "vehicles": vehicle_ids,
        "inputs": {
            "image_root": str(image_root),
            "passes_file": str(passes_file),
            "temperature_file": None if temperature_file is None else str(temperature_file),
            "temperature_target_year": temperature_target_year,
            "distance_file": None if distance_file is None else str(distance_file),
            "metadata_file": None if metadata_file is None else str(metadata_file),
            "feedback_file": None if feedback_file is None else str(feedback_file),
            "route_pattern_clusters": route_pattern_clusters,
            "behavior_clusters": behavior_clusters,
            "stop_profile_weight": stop_profile_weight,
            "scenario_context_start": scenario_context_start,
            "scenario_context_end": scenario_context_end,
            "use_validation_feedback": use_validation_feedback,
        },
        "closed_loop_design": {
            "step_1": "Split 2023 data by date so future days do not leak into training.",
            "step_2": "Build a truck-stop correlation graph from stop frequencies and truck purpose, then cluster trucks by behavioral similarity.",
            "step_3": "Cluster each truck's training route images and weight route patterns by that truck's own pattern frequency.",
            "step_4": "Give behavioral cluster summaries and route-pattern priors to the LLM for scenario generation.",
            "step_5": "Use each truck's prior-weighted route-pattern distance as the generated scenario route-distance expectation.",
            "step_6": "Validate the full multi-truck scenario against held-out real per-truck and fleet-total distances.",
            "step_7": "Learn feature weights from feedback when feedback labels are available, then score validation and test periods.",
            "step_8": "Append new feedback and rerun to correct truck correlation over time.",
        },
        "feature_meaning": {
            "pass_score": "Pearson pass correlation rescaled from [-1, 1] to [0, 1].",
            "image_similarity": "Average same-date cosine similarity of trajectory image features.",
            "distance_similarity": "Similarity of same-date real daily distances, combining total distance match, daily distance match, and distance-pattern correlation.",
            "purpose_match": "1 when truck purpose matches, 0 when it differs; omitted until metadata is provided.",
            "temperature_similarity": "Similarity of average temperatures observed by each truck in the split; omitted until temperature data is provided.",
        },
        "performance_goal": (
            "Closed-loop performance is evaluated by how well generated scenario "
            "route distances resemble real held-out route behavior for the full "
            "set of trucks. Pairwise distance similarity and train stability are "
            "retained as secondary diagnostics."
        ),
        "truck_stop_behavior_graph": truck_stop_behavior_graph,
        "route_pattern_expectations": route_pattern_expectations,
        "dominant_cluster_choices": dominant_cluster_choices,
        "recent_behavior_context": recent_behavior_context,
        "recent_context_cluster_choices": recent_context_cluster_choices,
        "validation_feedback_context": validation_feedback_context,
        "validation_feedback_cluster_choices": validation_feedback_cluster_choices,
        "learned_weights": learned_weights,
        "feedback_pairs_used": len(feedback),
        "splits": split_results,
        "performance_vs_train": _build_performance_vs_train(split_results),
        "scenario_realism_validation": _build_scenario_realism_validation(
            vehicle_ids,
            distances_by_vehicle,
            splits,
            route_pattern_expectations,
        ),
        "dominant_cluster_choice_realism_validation": _build_cluster_choice_realism_validation(
            vehicle_ids,
            distances_by_vehicle,
            splits,
            dominant_cluster_choices,
        ),
        "recent_context_cluster_choice_realism_validation": _build_cluster_choice_realism_validation(
            vehicle_ids,
            distances_by_vehicle,
            splits,
            recent_context_cluster_choices,
        ),
        "validation_feedback_cluster_choice_realism_validation": (
            _build_cluster_choice_realism_validation(
                vehicle_ids,
                distances_by_vehicle,
                splits,
                validation_feedback_cluster_choices,
            )
            if validation_feedback_cluster_choices is not None
            else None
        ),
        "real_route_distance_validation": _build_real_route_validation(split_results),
    }

    if use_llm:
        run["_runtime_distances_by_vehicle"] = distances_by_vehicle
        run = _apply_llm_weight_update(run, llm_model, llm_provider)
        run.pop("_runtime_distances_by_vehicle", None)

    output_dir = output_root / ("vehicles_" + "_".join(str(vehicle_id) for vehicle_id in vehicle_ids))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "closed_loop_run.json"
    output_file.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    run["output_file"] = str(output_file)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop truck-correlation agent design run.")
    parser.add_argument("vehicles", nargs="+", type=int)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--passes-file", type=Path, default=DEFAULT_PASSES_FILE)
    parser.add_argument("--temperature-file", type=Path)
    parser.add_argument("--temperature-target-year", type=int)
    parser.add_argument("--distance-file", type=Path)
    parser.add_argument("--metadata-file", type=Path)
    parser.add_argument("--feedback-file", type=Path)
    parser.add_argument("--train-start", default="2023-01-01")
    parser.add_argument("--train-end", default="2023-10-31")
    parser.add_argument("--validation-start", default="2023-11-01")
    parser.add_argument("--validation-end", default="2023-11-30")
    parser.add_argument("--test-start", default="2023-12-01")
    parser.add_argument("--test-end", default="2023-12-31")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--route-pattern-clusters", type=int, default=DEFAULT_ROUTE_PATTERN_CLUSTERS)
    parser.add_argument("--behavior-clusters", type=int, default=DEFAULT_BEHAVIOR_CLUSTERS)
    parser.add_argument("--stop-profile-weight", type=float, default=DEFAULT_STOP_PROFILE_WEIGHT)
    parser.add_argument("--scenario-context-start")
    parser.add_argument("--scenario-context-end")
    parser.add_argument("--use-validation-feedback", action="store_true")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-provider", choices=("anthropic", "openai"), default="anthropic")
    parser.add_argument("--llm-model")
    args = parser.parse_args()

    run = build_closed_loop_run(
        args.vehicles,
        image_root=args.image_root,
        passes_file=args.passes_file,
        temperature_file=args.temperature_file,
        temperature_target_year=args.temperature_target_year,
        distance_file=args.distance_file,
        metadata_file=args.metadata_file,
        feedback_file=args.feedback_file,
        train_start=args.train_start,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start,
        test_end=args.test_end,
        output_root=args.output_root,
        route_pattern_clusters=args.route_pattern_clusters,
        behavior_clusters=args.behavior_clusters,
        stop_profile_weight=args.stop_profile_weight,
        scenario_context_start=args.scenario_context_start,
        scenario_context_end=args.scenario_context_end,
        use_validation_feedback=args.use_validation_feedback,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
    )

    print(f"Wrote closed-loop run to {run['output_file']}")
    print(f"Learned weights: {run['learned_weights']}")
    print("behavioral clusters:")
    for cluster in run["truck_stop_behavior_graph"]["behavioral_clusters"]:
        print(
            f"  Cluster {cluster['cluster']}: vehicles={cluster['vehicles']}, "
            f"purposes={cluster['purpose_counts']}"
        )
    for split_name, split in run["splits"].items():
        print(f"{split_name}: {split['start_date']} to {split['end_date']}")
        for row in split["pairwise_relationships"]:
            score = row["learned_similarity"]
            score_text = "n/a" if score is None else f"{score:.3f}"
            print(
                f"  Vehicle {row['vehicle_a']} + {row['vehicle_b']}: "
                f"score={score_text}, {row['scenario_label']}"
            )
    print("performance vs training:")
    for split_name, performance in run["performance_vs_train"].items():
        delta = performance["mean_absolute_delta_from_train"]
        delta_text = "n/a" if delta is None else f"{delta:.3f}"
        print(f"  {split_name}: mean absolute delta from train={delta_text}")
    print("scenario realism validation:")
    for split_name in ("validation", "test"):
        validation = run["scenario_realism_validation"][split_name]
        error = validation["scenario_realism_error"]
        truck_error = validation["mean_per_truck_relative_distance_error"]
        fleet_error = validation["fleet_total_relative_distance_error"]
        error_text = "n/a" if error is None else f"{error:.3f}"
        truck_text = "n/a" if truck_error is None else f"{truck_error:.3f}"
        fleet_text = "n/a" if fleet_error is None else f"{fleet_error:.3f}"
        print(
            f"  {split_name}: scenario_error={error_text}, "
            f"truck_error={truck_text}, fleet_error={fleet_text}"
        )
    print("dominant cluster choice validation:")
    for split_name in ("validation", "test"):
        validation = run["dominant_cluster_choice_realism_validation"][split_name]
        error = validation["scenario_realism_error"]
        error_text = "n/a" if error is None else f"{error:.3f}"
        print(f"  {split_name}: scenario_error={error_text}")
    if run.get("recent_behavior_context") is not None:
        print("recent-context cluster choice validation:")
        for split_name in ("validation", "test"):
            validation = run["recent_context_cluster_choice_realism_validation"][split_name]
            error = validation["scenario_realism_error"]
            error_text = "n/a" if error is None else f"{error:.3f}"
            print(f"  {split_name}: scenario_error={error_text}")
    if run.get("validation_feedback_cluster_choice_realism_validation") is not None:
        print("validation-feedback cluster choice validation:")
        for split_name in ("validation", "test"):
            validation = run["validation_feedback_cluster_choice_realism_validation"][split_name]
            error = validation["scenario_realism_error"]
            error_text = "n/a" if error is None else f"{error:.3f}"
            print(f"  {split_name}: scenario_error={error_text}")
    print("real route distance validation:")
    for split_name, validation in run["real_route_distance_validation"].items():
        error = validation["mean_absolute_error_vs_real_distance"]
        error_text = "n/a" if error is None else f"{error:.3f}"
        print(f"  {split_name}: mean absolute error vs real distance={error_text}")
    if "llm_weight_update" in run:
        print(f"LLM updated weights: {run['llm_weight_update']['updated_weights']}")
        print("LLM performance vs training:")
        for split_name, performance in run["llm_performance_vs_train"].items():
            delta = performance["mean_absolute_delta_from_train"]
            delta_text = "n/a" if delta is None else f"{delta:.3f}"
            print(f"  {split_name}: mean absolute delta from train={delta_text}")
        print("LLM real route distance validation:")
        for split_name, validation in run["llm_real_route_distance_validation"].items():
            error = validation["mean_absolute_error_vs_real_distance"]
            error_text = "n/a" if error is None else f"{error:.3f}"
            print(f"  {split_name}: mean absolute error vs real distance={error_text}")
        print("LLM cluster choice validation:")
        for split_name, validation in run["llm_cluster_choice_realism_validation"].items():
            if split_name not in ("validation", "test"):
                continue
            error = validation["scenario_realism_error"]
            error_text = "n/a" if error is None else f"{error:.3f}"
            print(f"  {split_name}: scenario_error={error_text}")


if __name__ == "__main__":
    main()
