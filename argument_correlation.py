import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from correlation_calculator import (
    DEFAULT_PASSES_FILE,
    _load_stop_profiles,
    _normalize_date,
    _pearson,
)
from data_representation import (
    DEFAULT_FLEET_FILE,
    DEFAULT_PURPOSE_FILE,
    _load_fleet_metadata,
    _load_purpose,
)
from image_cluster import (
    DEFAULT_CLUSTERS,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_IMAGE_SIZE,
    cluster_vehicle_images,
)


DEFAULT_OUTPUT_DIR = Path("argument_correlation")
DEFAULT_WEIGHTS = {
    "stop_profile_correlation": 0.45,
    "route_similarity": 0.15,
    "purpose_match": 0.12,
    "age_similarity": 0.08,
    "kms_similarity": 0.08,
    "duty_cycle_match": 0.12,
}


def _numeric_similarity(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    denominator = max(abs(left), abs(right), 1e-8)
    if denominator == 1e-8:
        return 1.0
    return 1.0 - min(abs(left - right) / denominator, 1.0)


def _categorical_match(left: object, right: object) -> Optional[float]:
    if left in ("", None) or right in ("", None):
        return None
    return 1.0 if str(left).strip() == str(right).strip() else 0.0


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {
        key: max(0.0, float(weights.get(key, 0.0)))
        for key in DEFAULT_WEIGHTS
    }
    total = sum(cleaned.values())
    if total == 0:
        return dict(DEFAULT_WEIGHTS)
    return {key: value / total for key, value in cleaned.items()}


def _weighted_sum(features: dict[str, Optional[float]], weights: dict[str, float]) -> Optional[float]:
    weighted_values = [
        (float(value), weights[key])
        for key, value in features.items()
        if value is not None and weights.get(key, 0.0) > 0
    ]
    if not weighted_values:
        return None

    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight == 0:
        return None
    return sum(value * weight for value, weight in weighted_values) / total_weight


def _cluster_result_for_vehicle(
    vehicle_id: int,
    start_date: str,
    end_date: str,
    image_cluster_results: Optional[dict],
    image_root: Path,
    output_dir: Path,
    feature_method: str,
    n_clusters: int,
    image_size: int,
) -> Optional[dict]:
    if image_cluster_results:
        if str(vehicle_id) in image_cluster_results:
            return image_cluster_results[str(vehicle_id)]
        if vehicle_id in image_cluster_results:
            return image_cluster_results[vehicle_id]

    try:
        return cluster_vehicle_images(
            vehicle_id=vehicle_id,
            start_date=start_date,
            end_date=end_date,
            image_root=image_root,
            output_dir=output_dir,
            feature_method=feature_method,
            n_clusters=n_clusters,
            image_size=image_size,
        )
    except ValueError:
        return None


def _route_ink_vector(image_path: str, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("L").resize((image_size, image_size))
    vector = 1.0 - (np.asarray(image, dtype=np.float32).reshape(-1) / 255.0)
    norm = np.linalg.norm(vector)
    if norm <= 1e-8:
        return vector
    return vector / norm


def _representative_route_vectors(cluster_result: Optional[dict], image_size: int) -> list[dict]:
    if not cluster_result:
        return []

    rows_by_date = {
        row["date"]: row
        for row in cluster_result.get("clusters_by_date", [])
    }
    vectors = []
    for cluster in cluster_result.get("cluster_summary", []):
        representative_date = cluster.get("representative_date")
        row = rows_by_date.get(representative_date)
        image_path = row.get("image_path") if row else None
        if not image_path:
            continue
        path = Path(image_path)
        if not path.exists():
            continue
        vectors.append(
            {
                "cluster": cluster.get("cluster"),
                "representative_date": representative_date,
                "image_path": str(path),
                "share": float(cluster.get("share") or 0.0),
                "vector": _route_ink_vector(str(path), image_size),
            }
        )
    return vectors


def _route_similarity_from_representatives(left: list[dict], right: list[dict]) -> Optional[float]:
    if not left or not right:
        return None

    weighted_similarity = 0.0
    total_weight = 0.0
    for left_route in left:
        for right_route in right:
            weight = max(0.0, left_route["share"]) * max(0.0, right_route["share"])
            if weight == 0:
                continue
            similarity = float(np.dot(left_route["vector"], right_route["vector"]))
            similarity = max(0.0, min(1.0, similarity))
            weighted_similarity += similarity * weight
            total_weight += weight

    if total_weight == 0:
        return None
    return weighted_similarity / total_weight


def _parse_weight_overrides(values: list[str]) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"Weight override must use key=value format: {value}")
        key, raw_weight = value.split("=", 1)
        if key not in DEFAULT_WEIGHTS:
            allowed = ", ".join(DEFAULT_WEIGHTS)
            raise ValueError(f"Unknown weight key {key!r}. Allowed keys: {allowed}")
        weights[key] = float(raw_weight)
    return _normalize_weights(weights)


def calculate_statistical_argument_correlation(
    vehicle_ids: list[int],
    start_date: str,
    end_date: str,
    passes_file: Path = DEFAULT_PASSES_FILE,
    purpose_file: Path = DEFAULT_PURPOSE_FILE,
    fleet_file: Path = DEFAULT_FLEET_FILE,
    weights: Optional[dict[str, float]] = None,
    image_cluster_results: Optional[dict] = None,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    route_cluster_output_dir: Path = DEFAULT_OUTPUT_DIR / "route_clusters",
    route_feature_method: str = "pca",
    route_clusters: int = DEFAULT_CLUSTERS,
    route_image_size: int = DEFAULT_IMAGE_SIZE,
) -> dict:
    normalized_start = _normalize_date(start_date)
    normalized_end = _normalize_date(end_date)
    if normalized_start > normalized_end:
        raise ValueError("--start-date must be before or equal to --end-date.")

    resolved_weights = _normalize_weights(weights or DEFAULT_WEIGHTS)
    profiles, zone_columns = _load_stop_profiles(
        passes_file,
        vehicle_ids,
        normalized_start,
        normalized_end,
    )
    purposes = _load_purpose(purpose_file)
    fleet_metadata = _load_fleet_metadata(fleet_file)
    route_clusters_by_vehicle = {
        vehicle_id: _cluster_result_for_vehicle(
            vehicle_id=vehicle_id,
            start_date=normalized_start,
            end_date=normalized_end,
            image_cluster_results=image_cluster_results,
            image_root=image_root,
            output_dir=route_cluster_output_dir,
            feature_method=route_feature_method,
            n_clusters=route_clusters,
            image_size=route_image_size,
        )
        for vehicle_id in vehicle_ids
    }
    representative_routes = {
        vehicle_id: _representative_route_vectors(cluster_result, route_image_size)
        for vehicle_id, cluster_result in route_clusters_by_vehicle.items()
    }

    rows = []
    for index, vehicle_a in enumerate(vehicle_ids):
        for vehicle_b in vehicle_ids[index + 1 :]:
            profile_a = profiles[vehicle_a]
            profile_b = profiles[vehicle_b]
            fleet_a = fleet_metadata.get(vehicle_a, {})
            fleet_b = fleet_metadata.get(vehicle_b, {})

            features = {
                "stop_profile_correlation": _pearson(profile_a.vector, profile_b.vector),
                "route_similarity": _route_similarity_from_representatives(
                    representative_routes[vehicle_a],
                    representative_routes[vehicle_b],
                ),
                "purpose_match": _categorical_match(purposes.get(vehicle_a), purposes.get(vehicle_b)),
                "age_similarity": _numeric_similarity(fleet_a.get("age"), fleet_b.get("age")),
                "kms_similarity": _numeric_similarity(fleet_a.get("kms"), fleet_b.get("kms")),
                "duty_cycle_match": _categorical_match(fleet_a.get("duty_cycle"), fleet_b.get("duty_cycle")),
            }
            total_correlation = _weighted_sum(features, resolved_weights)

            rows.append(
                {
                    "vehicle_a": vehicle_a,
                    "vehicle_b": vehicle_b,
                    "start_date": normalized_start,
                    "end_date": normalized_end,
                    "method": "statistical",
                    "total_correlation": total_correlation,
                    "features": features,
                    "weights": resolved_weights,
                    "vehicle_a_observed_days": profile_a.observed_days,
                    "vehicle_b_observed_days": profile_b.observed_days,
                }
            )

    rows.sort(
        key=lambda row: row["total_correlation"] if row["total_correlation"] is not None else -2.0,
        reverse=True,
    )

    return {
        "method": {
            "name": "statistical",
            "description": (
                "Compute stop-profile vector correlation, append fleet metadata feature "
                "similarities and representative route-image similarity by weight, "
                "and sum them into total_correlation."
            ),
            "period": {"start_date": normalized_start, "end_date": normalized_end},
            "zone_columns": zone_columns,
            "vector_order": next(iter(profiles.values())).vector_labels if profiles else [],
            "weights": resolved_weights,
            "route_similarity": {
                "description": (
                    "For each truck, choose the representative image from each route "
                    "cluster, compare representative route-ink vectors by cosine "
                    "similarity, and average pairwise similarities weighted by cluster shares."
                ),
                "feature_method": route_feature_method,
                "route_clusters": route_clusters,
                "image_size": route_image_size,
            },
        },
        "profiles": {
            str(vehicle_id): {
                "vehicle_id": profile.vehicle_id,
                "observed_days": profile.observed_days,
                "vector_labels": profile.vector_labels,
                "vector": profile.vector,
            }
            for vehicle_id, profile in profiles.items()
        },
        "fleet_metadata": {
            str(vehicle_id): {
                "purpose": purposes.get(vehicle_id),
                "age": fleet_metadata.get(vehicle_id, {}).get("age"),
                "kms": fleet_metadata.get(vehicle_id, {}).get("kms"),
                "duty_cycle": fleet_metadata.get(vehicle_id, {}).get("duty_cycle"),
            }
            for vehicle_id in vehicle_ids
        },
        "representative_routes": {
            str(vehicle_id): [
                {
                    "cluster": route["cluster"],
                    "representative_date": route["representative_date"],
                    "image_path": route["image_path"],
                    "share": route["share"],
                }
                for route in routes
            ]
            for vehicle_id, routes in representative_routes.items()
        },
        "pairwise_argument_correlations": rows,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "vehicle_a",
        "vehicle_b",
        "start_date",
        "end_date",
        "method",
        "total_correlation",
        "stop_profile_correlation",
        "route_similarity",
        "purpose_match",
        "age_similarity",
        "kms_similarity",
        "duty_cycle_match",
        "vehicle_a_observed_days",
        "vehicle_b_observed_days",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = {
                "vehicle_a": row["vehicle_a"],
                "vehicle_b": row["vehicle_b"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "method": row["method"],
                "total_correlation": row["total_correlation"],
                "vehicle_a_observed_days": row["vehicle_a_observed_days"],
                "vehicle_b_observed_days": row["vehicle_b_observed_days"],
                **row["features"],
            }
            for key, value in list(output_row.items()):
                if isinstance(value, float):
                    output_row[key] = f"{value:.6f}"
            writer.writerow(output_row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate argument correlations with the statistical method: weighted "
            "stop-profile correlation plus fleet metadata similarities."
        )
    )
    parser.add_argument("vehicles", nargs="+", type=int, help="Vehicle IDs to compare.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--passes-file", type=Path, default=DEFAULT_PASSES_FILE)
    parser.add_argument("--purpose-file", type=Path, default=DEFAULT_PURPOSE_FILE)
    parser.add_argument("--fleet-file", type=Path, default=DEFAULT_FLEET_FILE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--route-cluster-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "route_clusters")
    parser.add_argument("--route-feature-method", choices=("pca", "autoencoder"), default="pca")
    parser.add_argument("--route-clusters", type=int, default=DEFAULT_CLUSTERS)
    parser.add_argument("--route-image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help=(
            "Override a statistical weight as key=value. Allowed keys: "
            + ", ".join(DEFAULT_WEIGHTS)
        ),
    )
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    result = calculate_statistical_argument_correlation(
        args.vehicles,
        args.start_date,
        args.end_date,
        passes_file=args.passes_file,
        purpose_file=args.purpose_file,
        fleet_file=args.fleet_file,
        weights=_parse_weight_overrides(args.weight),
        image_root=args.image_root,
        route_cluster_output_dir=args.route_cluster_output_dir,
        route_feature_method=args.route_feature_method,
        route_clusters=args.route_clusters,
        route_image_size=args.route_image_size,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    start_date = result["method"]["period"]["start_date"]
    end_date = result["method"]["period"]["end_date"]
    vehicle_label = "_".join(str(vehicle_id) for vehicle_id in args.vehicles)
    base_name = f"statistical_period_{start_date}_{end_date}_vehicles_{vehicle_label}"
    json_file = output_dir / f"{base_name}.json"
    csv_file = output_dir / f"{base_name}.csv"

    json_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_file, result["pairwise_argument_correlations"])

    print(f"Wrote full result to {json_file}")
    print(f"Wrote pairwise argument correlations to {csv_file}")
    print("Top argument correlations:")
    for row in result["pairwise_argument_correlations"][: args.top]:
        score = row["total_correlation"]
        score_text = "n/a" if score is None else f"{score:.3f}"
        print(
            f"  Vehicle {row['vehicle_a']} + {row['vehicle_b']}: "
            f"total_correlation={score_text}, "
            f"stop_profile={row['features']['stop_profile_correlation']:.3f}"
        )


if __name__ == "__main__":
    main()
