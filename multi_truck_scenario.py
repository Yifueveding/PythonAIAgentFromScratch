import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


DEFAULT_IMAGE_ROOT = Path("Vehicle")
DEFAULT_PASSES_FILE = Path("vehicle_day_zone_passes.csv")
IMAGE_SIZE = 32


def _vehicle_image_dir(image_root: Path, vehicle_id: int) -> Path:
    nested = image_root / f"Vehicle_{vehicle_id}"
    if nested.exists():
        return nested
    return Path(f"Vehicle_{vehicle_id}")


def _image_to_feature(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    pixels = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
    route_ink = 1.0 - pixels
    norm = np.linalg.norm(route_ink)
    if norm == 0:
        return route_ink
    return route_ink / norm


def _load_image_features(image_root: Path, vehicle_id: int) -> dict[str, np.ndarray]:
    vehicle_dir = _vehicle_image_dir(image_root, vehicle_id)
    image_paths = sorted(vehicle_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found for Vehicle {vehicle_id} in {vehicle_dir}.")

    return {path.stem: _image_to_feature(path) for path in image_paths}


def _fit_pca(features: np.ndarray, n_components: int) -> np.ndarray:
    centered = features - features.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    transformed = centered @ vt[:n_components].T
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    return transformed / np.maximum(norms, 1e-8)


def _fit_kmeans(features: np.ndarray, n_clusters: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    initial_indices = np.linspace(0, len(features) - 1, n_clusters, dtype=int)
    centroids = features[initial_indices].copy()

    for _ in range(max_iter):
        distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        next_centroids = centroids.copy()

        for cluster_id in range(n_clusters):
            members = features[labels == cluster_id]
            if len(members):
                next_centroids[cluster_id] = members.mean(axis=0)

        if np.allclose(next_centroids, centroids):
            break
        centroids = next_centroids

    distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
    return distances.argmin(axis=1), centroids


def _cluster_vehicle_images(
    image_root: Path,
    vehicle_id: int,
    requested_clusters: int,
) -> dict:
    image_features = _load_image_features(image_root, vehicle_id)
    dates = sorted(image_features)
    raw_features = np.vstack([image_features[date] for date in dates])

    n_clusters = min(requested_clusters, len(dates))
    if n_clusters < 2:
        labels = np.zeros(len(dates), dtype=int)
        distances = np.zeros(len(dates), dtype=float)
    else:
        n_components = min(21, len(dates) - 1, raw_features.shape[1])
        features = _fit_pca(raw_features, n_components)
        labels, centroids = _fit_kmeans(features, n_clusters)
        distances = np.linalg.norm(features - centroids[labels], axis=1)

    cluster_counts = Counter(int(label) for label in labels)
    cluster_rows = []
    for cluster_id, count in cluster_counts.most_common():
        cluster_dates = [date for date, label in zip(dates, labels) if int(label) == cluster_id]
        cluster_rows.append(
            {
                "cluster": cluster_id,
                "days": count,
                "share": count / len(dates),
                "example_dates": cluster_dates[:5],
            }
        )

    return {
        "vehicle_id": vehicle_id,
        "image_count": len(dates),
        "dates": dates,
        "features_by_date": image_features,
        "labels_by_date": {date: int(label) for date, label in zip(dates, labels)},
        "distances_by_date": {date: float(distance) for date, distance in zip(dates, distances)},
        "cluster_summary": cluster_rows,
    }


def _load_vehicle_passes(path: Path) -> tuple[dict[int, dict[str, tuple[int, ...]]], list[str]]:
    vehicles: dict[int, dict[str, tuple[int, ...]]] = defaultdict(dict)

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")

        zone_columns = [column for column in reader.fieldnames if column.startswith("passes_zone_")]
        if not zone_columns:
            raise ValueError(f"{path} does not contain passes_zone_* columns.")

        for row in reader:
            vehicle_id = int(row["VehicleId"])
            vehicles[vehicle_id][row["date"]] = tuple(int(row[column]) for column in zone_columns)

    return dict(vehicles), zone_columns


def _pearson(left: list[float], right: list[float]) -> Optional[float]:
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
    return numerator / denominator


def _cosine(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return None
    return float(np.dot(left, right) / denominator)


def _pass_correlation(
    left: dict[str, tuple[int, ...]],
    right: dict[str, tuple[int, ...]],
) -> tuple[Optional[float], int, int]:
    common_dates = sorted(set(left) & set(right))
    left_values: list[float] = []
    right_values: list[float] = []
    for date in common_dates:
        left_values.extend(left[date])
        right_values.extend(right[date])

    return _pearson(left_values, right_values), len(common_dates), len(left_values)


def _image_similarity(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> tuple[Optional[float], int, list[str]]:
    common_dates = sorted(set(left) & set(right))
    similarities = []
    for date in common_dates:
        similarity = _cosine(left[date], right[date])
        if similarity is not None:
            similarities.append((date, similarity))

    if not similarities:
        return None, 0, []

    mean_similarity = sum(similarity for _, similarity in similarities) / len(similarities)
    top_dates = [
        date
        for date, _ in sorted(similarities, key=lambda item: item[1], reverse=True)[:5]
    ]
    return mean_similarity, len(similarities), top_dates


def _combined_similarity(pass_corr: Optional[float], image_similarity: Optional[float], pass_weight: float) -> Optional[float]:
    weighted_values = []
    if pass_corr is not None:
        weighted_values.append((max(0.0, min(1.0, (pass_corr + 1.0) / 2.0)), pass_weight))
    if image_similarity is not None:
        weighted_values.append((max(0.0, min(1.0, image_similarity)), 1.0 - pass_weight))
    if not weighted_values:
        return None

    total_weight = sum(weight for _, weight in weighted_values)
    return sum(value * weight for value, weight in weighted_values) / total_weight


def _summarize_pass_zones(
    passes: dict[str, tuple[int, ...]],
    zone_columns: list[str],
) -> list[dict]:
    if not passes:
        return []

    totals = [0] * len(zone_columns)
    for values in passes.values():
        for index, value in enumerate(values):
            totals[index] += value

    rows = []
    for zone, total in zip(zone_columns, totals):
        rows.append({"zone": zone, "days": total, "share": total / len(passes)})
    rows.sort(key=lambda row: row["share"], reverse=True)
    return rows


def _scenario_label(score: Optional[float]) -> str:
    if score is None:
        return "insufficient_overlap"
    if score >= 0.75:
        return "shared_route_behavior"
    if score >= 0.55:
        return "partially_shared_route_behavior"
    return "mostly_independent_route_behavior"


def build_multi_truck_scenario(
    vehicle_ids: list[int],
    image_root: Path = DEFAULT_IMAGE_ROOT,
    passes_file: Path = DEFAULT_PASSES_FILE,
    n_clusters: int = 10,
    pass_weight: float = 0.5,
) -> dict:
    passes_by_vehicle, zone_columns = _load_vehicle_passes(passes_file)
    image_models = {
        vehicle_id: _cluster_vehicle_images(image_root, vehicle_id, n_clusters)
        for vehicle_id in vehicle_ids
    }

    vehicle_summaries = []
    for vehicle_id in vehicle_ids:
        pass_summary = _summarize_pass_zones(passes_by_vehicle.get(vehicle_id, {}), zone_columns)
        cluster_summary = image_models[vehicle_id]["cluster_summary"]
        top_cluster = cluster_summary[0] if cluster_summary else None
        vehicle_summaries.append(
            {
                "vehicle_id": vehicle_id,
                "image_days": image_models[vehicle_id]["image_count"],
                "pass_point_days": len(passes_by_vehicle.get(vehicle_id, {})),
                "dominant_local_cluster": top_cluster,
                "alternate_local_clusters": cluster_summary[1:4],
                "frequent_pass_zones": pass_summary[:3],
            }
        )

    pairwise = []
    for left_id, right_id in combinations(vehicle_ids, 2):
        pass_corr, common_pass_dates, pass_features = _pass_correlation(
            passes_by_vehicle.get(left_id, {}),
            passes_by_vehicle.get(right_id, {}),
        )
        image_sim, common_image_dates, similar_dates = _image_similarity(
            image_models[left_id]["features_by_date"],
            image_models[right_id]["features_by_date"],
        )
        combined = _combined_similarity(pass_corr, image_sim, pass_weight)
        pairwise.append(
            {
                "vehicle_a": left_id,
                "vehicle_b": right_id,
                "pass_correlation": pass_corr,
                "image_similarity": image_sim,
                "combined_similarity": combined,
                "scenario_label": _scenario_label(combined),
                "common_pass_dates": common_pass_dates,
                "pass_features_compared": pass_features,
                "common_image_dates": common_image_dates,
                "most_similar_image_dates": similar_dates,
            }
        )

    pairwise.sort(
        key=lambda row: row["combined_similarity"] if row["combined_similarity"] is not None else -1.0,
        reverse=True,
    )

    return {
        "vehicles": vehicle_ids,
        "method": {
            "local_clusters": "Each truck is clustered separately to capture that truck's own route-pattern frequency.",
            "pass_similarity": "Pearson correlation of shared daily passes_zone_* vectors.",
            "image_similarity": "Average same-date cosine similarity of trajectory image features.",
            "combined_similarity": f"{pass_weight:.2f} pass score + {1.0 - pass_weight:.2f} image score.",
        },
        "vehicle_summaries": vehicle_summaries,
        "pairwise_relationships": pairwise,
        "group_scenarios": _build_group_scenarios(vehicle_summaries, pairwise),
    }


def _build_group_scenarios(vehicle_summaries: list[dict], pairwise: list[dict]) -> list[dict]:
    scenarios = []

    scenarios.append(
        {
            "name": "regular_behavior",
            "description": "Use each truck's dominant local cluster as its normal route pattern.",
            "evidence": [
                {
                    "vehicle_id": summary["vehicle_id"],
                    "dominant_local_cluster": summary["dominant_local_cluster"],
                    "frequent_pass_zones": summary["frequent_pass_zones"],
                }
                for summary in vehicle_summaries
            ],
        }
    )

    shared_pairs = [
        row
        for row in pairwise
        if row["combined_similarity"] is not None and row["combined_similarity"] >= 0.55
    ]
    scenarios.append(
        {
            "name": "shared_or_substitutable_routes",
            "description": "Truck pairs with higher combined similarity may represent shared route behavior or substitute route capacity.",
            "evidence": shared_pairs,
        }
    )

    scenarios.append(
        {
            "name": "alternate_or_exception_routes",
            "description": "Less frequent local clusters are candidate alternate, seasonal, or exception route patterns for each truck.",
            "evidence": [
                {
                    "vehicle_id": summary["vehicle_id"],
                    "alternate_local_clusters": summary["alternate_local_clusters"],
                }
                for summary in vehicle_summaries
            ],
        }
    )

    return scenarios


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate route scenarios for multiple trucks.")
    parser.add_argument("vehicles", nargs="+", type=int, help="Vehicle IDs to analyze together.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--passes-file", type=Path, default=DEFAULT_PASSES_FILE)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--pass-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not 0.0 <= args.pass_weight <= 1.0:
        raise ValueError("--pass-weight must be between 0.0 and 1.0.")

    scenario = build_multi_truck_scenario(
        args.vehicles,
        image_root=args.image_root,
        passes_file=args.passes_file,
        n_clusters=args.clusters,
        pass_weight=args.pass_weight,
    )

    output = args.output or Path("multi_truck_scenarios_" + "_".join(str(v) for v in args.vehicles) + ".json")
    _write_json(output, scenario)

    print(f"Wrote scenario analysis to {output}")
    print("Pairwise truck relationships:")
    for row in scenario["pairwise_relationships"]:
        combined = row["combined_similarity"]
        combined_text = "n/a" if combined is None else f"{combined:.3f}"
        pass_text = "n/a" if row["pass_correlation"] is None else f"{row['pass_correlation']:.3f}"
        image_text = "n/a" if row["image_similarity"] is None else f"{row['image_similarity']:.3f}"
        print(
            f"Vehicle {row['vehicle_a']} + {row['vehicle_b']}: "
            f"combined={combined_text}, pass_corr={pass_text}, image_sim={image_text}, "
            f"{row['scenario_label']}"
        )


if __name__ == "__main__":
    main()
