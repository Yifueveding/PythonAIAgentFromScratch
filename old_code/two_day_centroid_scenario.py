import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_IMAGE_ROOT = Path("Vehicle")
DEFAULT_PASSES_FILE = Path("vehicle_day_zone_passes.csv")
DEFAULT_OUTPUT_ROOT = Path("two_day_scenarios")
IMAGE_SIZE = 32


def _vehicle_image_dir(image_root: Path, vehicle_id: int) -> Path:
    nested = image_root / f"Vehicle_{vehicle_id}"
    if nested.exists():
        return nested
    return Path(f"Vehicle_{vehicle_id}")


def _image_path(image_root: Path, vehicle_id: int, date: str) -> Path:
    return _vehicle_image_dir(image_root, vehicle_id) / f"{date}.png"


def _image_to_pixels(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0


def _pixels_to_feature(pixels: np.ndarray) -> np.ndarray:
    route_ink = 1.0 - pixels
    norm = np.linalg.norm(route_ink)
    if norm == 0:
        return route_ink
    return route_ink / norm


def _load_images(image_root: Path, vehicle_id: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    vehicle_dir = _vehicle_image_dir(image_root, vehicle_id)
    image_paths = sorted(vehicle_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found for Vehicle {vehicle_id} in {vehicle_dir}.")

    dates = [path.stem for path in image_paths]
    pixels = np.vstack([_image_to_pixels(path) for path in image_paths])
    features = np.vstack([_pixels_to_feature(row) for row in pixels])
    return dates, pixels, features


def _fit_pca(features: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    centered = features - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    transformed = centered @ components.T
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    return mean, components, transformed / np.maximum(norms, 1e-8)


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


def _nearest_member_index(features: np.ndarray, labels: np.ndarray, cluster_id: int, centroid: np.ndarray) -> int:
    member_indices = np.where(labels == cluster_id)[0]
    member_features = features[member_indices]
    nearest_offset = np.linalg.norm(member_features - centroid, axis=1).argmin()
    return int(member_indices[nearest_offset])


def _write_centroid_image(path: Path, centroid_pixels: np.ndarray) -> None:
    pixels = np.clip(centroid_pixels.reshape(IMAGE_SIZE, IMAGE_SIZE) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def _load_square(path: Path, size: int = 160) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size))


def _write_pair_evidence_image(
    image_root: Path,
    vehicle_a: int,
    vehicle_b: int,
    date: str,
    output_path: Path,
) -> bool:
    left_path = _image_path(image_root, vehicle_a, date)
    right_path = _image_path(image_root, vehicle_b, date)
    if not left_path.exists() or not right_path.exists():
        return False

    left = _load_square(left_path)
    right = _load_square(right_path)
    label_height = 34
    gap = 12
    width = left.width + right.width + gap
    height = left.height + label_height
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(left, (0, label_height))
    canvas.paste(right, (left.width + gap, label_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((4, 8), f"Vehicle {vehicle_a}", fill="black")
    draw.text((left.width + gap + 4, 8), f"Vehicle {vehicle_b}", fill="black")
    draw.text((width // 2 - 42, height - 18), date, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return True


def _load_pass_zones(path: Path) -> tuple[dict[int, dict[str, tuple[int, ...]]], list[str]]:
    vehicles: dict[int, dict[str, tuple[int, ...]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        zone_columns = [column for column in reader.fieldnames if column.startswith("passes_zone_")]
        for row in reader:
            vehicles[int(row["VehicleId"])][row["date"]] = tuple(int(row[column]) for column in zone_columns)
    return dict(vehicles), zone_columns


def _zone_summary(
    vehicle_passes: dict[str, tuple[int, ...]],
    dates: list[str],
    zone_columns: list[str],
) -> list[dict]:
    totals = [0] * len(zone_columns)
    matching_dates = [date for date in dates if date in vehicle_passes]
    for date in matching_dates:
        for index, value in enumerate(vehicle_passes[date]):
            totals[index] += value

    denominator = max(1, len(matching_dates))
    rows = [
        {"zone": zone, "days": total, "share": total / denominator}
        for zone, total in zip(zone_columns, totals)
    ]
    rows.sort(key=lambda row: row["share"], reverse=True)
    return rows[:3]


def _cluster_vehicle(
    image_root: Path,
    vehicle_id: int,
    n_clusters: int,
    output_dir: Path,
    passes_by_vehicle: dict[int, dict[str, tuple[int, ...]]],
    zone_columns: list[str],
) -> dict:
    dates, pixels, image_features = _load_images(image_root, vehicle_id)
    cluster_count = min(n_clusters, len(dates))

    if cluster_count < 2:
        labels = np.zeros(len(dates), dtype=int)
        feature_space = image_features
        centroids = np.vstack([pixels.mean(axis=0)])
    else:
        n_components = min(21, len(dates) - 1, image_features.shape[1])
        _, _, feature_space = _fit_pca(image_features, n_components)
        labels, centroids = _fit_kmeans(feature_space, cluster_count)

    counts = Counter(int(label) for label in labels)
    ranked_clusters = [cluster_id for cluster_id, _ in counts.most_common()]
    selected_clusters = ranked_clusters[:2]

    vehicle_dir = output_dir / f"Vehicle_{vehicle_id}"
    vehicle_dir.mkdir(parents=True, exist_ok=True)

    scenario_days = []
    for scenario_index, cluster_id in enumerate(selected_clusters, start=1):
        nearest_index = _nearest_member_index(feature_space, labels, cluster_id, centroids[cluster_id])
        member_dates = [date for date, label in zip(dates, labels) if int(label) == cluster_id]
        centroid_pixels = pixels[[index for index, label in enumerate(labels) if int(label) == cluster_id]].mean(axis=0)

        centroid_file = vehicle_dir / f"day_{scenario_index}_cluster_{cluster_id}_centroid.png"
        representative_file = vehicle_dir / f"day_{scenario_index}_cluster_{cluster_id}_representative.png"
        _write_centroid_image(centroid_file, centroid_pixels)
        Image.open(_vehicle_image_dir(image_root, vehicle_id) / f"{dates[nearest_index]}.png").save(representative_file)

        scenario_days.append(
            {
                "scenario_day": scenario_index,
                "scenario_type": "regular_route" if scenario_index == 1 else "alternate_route",
                "local_cluster": cluster_id,
                "cluster_days": counts[cluster_id],
                "cluster_share": counts[cluster_id] / len(dates),
                "centroid_image": str(centroid_file),
                "representative_image": str(representative_file),
                "representative_date": dates[nearest_index],
                "example_dates": member_dates[:10],
                "frequent_pass_zones_for_cluster_dates": _zone_summary(
                    passes_by_vehicle.get(vehicle_id, {}),
                    member_dates,
                    zone_columns,
                ),
            }
        )

    return {
        "vehicle_id": vehicle_id,
        "image_days": len(dates),
        "scenario_days": scenario_days,
    }


def _relationship_text(row: dict) -> str:
    score = row.get("learned_similarity")
    image_similarity = row.get("features", {}).get("image_similarity")
    pass_corr = row.get("features", {}).get("pass_correlation")
    score_text = "n/a" if score is None else f"{score:.3f}"
    image_text = "n/a" if image_similarity is None else f"{image_similarity:.3f}"
    pass_text = "n/a" if pass_corr is None else f"{pass_corr:.3f}"
    return (
        f"score={score_text}, image_similarity={image_text}, "
        f"pass_correlation={pass_text}, label={row['scenario_label']}"
    )


def _add_closed_loop_evidence(
    scenario: dict,
    closed_loop_json: Path,
    image_root: Path,
    max_evidence_dates: int,
) -> None:
    run = json.loads(closed_loop_json.read_text(encoding="utf-8"))
    evidence_root = Path(scenario["output_dir"]) / "pairwise_evidence"
    evidence = {
        "source_closed_loop_run": str(closed_loop_json),
        "description": "Same-date side-by-side trajectory images for the most similar image dates in each closed-loop split.",
        "splits": {},
    }

    for split_name, split in run["splits"].items():
        split_rows = []
        for row in split["pairwise_relationships"]:
            vehicle_a = row["vehicle_a"]
            vehicle_b = row["vehicle_b"]
            pair_dir = evidence_root / split_name / f"Vehicle_{vehicle_a}_Vehicle_{vehicle_b}"
            pair_images = []

            for date in row.get("most_similar_image_dates", [])[:max_evidence_dates]:
                output_path = pair_dir / f"{date}.png"
                if _write_pair_evidence_image(image_root, vehicle_a, vehicle_b, date, output_path):
                    pair_images.append(str(output_path))

            split_rows.append(
                {
                    "vehicle_a": vehicle_a,
                    "vehicle_b": vehicle_b,
                    "relationship": _relationship_text(row),
                    "most_similar_image_dates": row.get("most_similar_image_dates", [])[:max_evidence_dates],
                    "pair_images": pair_images,
                }
            )

        evidence["splits"][split_name] = split_rows

    scenario["closed_loop_evidence"] = evidence


def _write_markdown(path: Path, scenario: dict) -> None:
    lines = [
        "# Two-Day Truck Scenario",
        "",
        "Day 1 uses each truck's most frequent local image cluster.",
        "Day 2 uses each truck's next most frequent local image cluster.",
        "",
    ]

    for vehicle in scenario["vehicles"]:
        lines.append(f"## Vehicle {vehicle['vehicle_id']}")
        lines.append("")
        for day in vehicle["scenario_days"]:
            zones = ", ".join(
                f"{zone['zone']} ({zone['days']} days)"
                for zone in day["frequent_pass_zones_for_cluster_dates"]
            )
            lines.extend(
                [
                    f"### Day {day['scenario_day']}: {day['scenario_type']}",
                    "",
                    f"- Local cluster: {day['local_cluster']}",
                    f"- Frequency: {day['cluster_days']} days ({day['cluster_share']:.1%})",
                    f"- Representative date: {day['representative_date']}",
                    f"- Centroid image: `{day['centroid_image']}`",
                    f"- Representative image: `{day['representative_image']}`",
                    f"- Frequent pass zones: {zones or 'none'}",
                    "",
                ]
            )

    evidence = scenario.get("closed_loop_evidence")
    if evidence:
        lines.extend(
            [
                "## Pairwise Closed-Loop Evidence",
                "",
                evidence["description"],
                "",
            ]
        )
        for split_name, rows in evidence["splits"].items():
            lines.extend([f"### {split_name}", ""])
            for row in rows:
                lines.extend(
                    [
                        f"#### Vehicle {row['vehicle_a']} + Vehicle {row['vehicle_b']}",
                        "",
                        row["relationship"],
                        "",
                    ]
                )
                if row["pair_images"]:
                    for image in row["pair_images"]:
                        lines.append(f"- `{image}`")
                else:
                    lines.append("- No same-date image evidence found.")
                lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_two_day_scenario(
    vehicle_ids: list[int],
    image_root: Path,
    passes_file: Path,
    output_root: Path,
    n_clusters: int,
    closed_loop_json: Optional[Path] = None,
    max_evidence_dates: int = 5,
) -> dict:
    output_dir = output_root / ("vehicles_" + "_".join(str(vehicle_id) for vehicle_id in vehicle_ids))
    output_dir.mkdir(parents=True, exist_ok=True)
    passes_by_vehicle, zone_columns = _load_pass_zones(passes_file)

    scenario = {
        "vehicles_requested": vehicle_ids,
        "method": {
            "day_1": "dominant local cluster centroid for each truck",
            "day_2": "second-most frequent local cluster centroid for each truck",
            "centroid": "average trajectory image for all dates assigned to that truck's local cluster",
            "representative_image": "real image closest to the cluster centroid in feature space",
        },
        "output_dir": str(output_dir),
        "vehicles": [
            _cluster_vehicle(
                image_root,
                vehicle_id,
                n_clusters,
                output_dir,
                passes_by_vehicle,
                zone_columns,
            )
            for vehicle_id in vehicle_ids
        ],
    }

    if closed_loop_json is not None:
        _add_closed_loop_evidence(
            scenario,
            closed_loop_json,
            image_root,
            max_evidence_dates,
        )

    json_path = output_dir / "two_day_scenario.json"
    markdown_path = output_dir / "README.md"
    json_path.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
    _write_markdown(markdown_path, scenario)
    scenario["json_file"] = str(json_path)
    scenario["markdown_file"] = str(markdown_path)
    return scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate two-day scenarios with local cluster centroids.")
    parser.add_argument("vehicles", nargs="+", type=int)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--passes-file", type=Path, default=DEFAULT_PASSES_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--closed-loop-json", type=Path)
    parser.add_argument("--max-evidence-dates", type=int, default=5)
    args = parser.parse_args()

    scenario = build_two_day_scenario(
        args.vehicles,
        image_root=args.image_root,
        passes_file=args.passes_file,
        output_root=args.output_root,
        n_clusters=args.clusters,
        closed_loop_json=args.closed_loop_json,
        max_evidence_dates=args.max_evidence_dates,
    )

    print(f"Wrote two-day scenario folder: {scenario['output_dir']}")
    print(f"JSON: {scenario['json_file']}")
    print(f"Summary: {scenario['markdown_file']}")


if __name__ == "__main__":
    main()
