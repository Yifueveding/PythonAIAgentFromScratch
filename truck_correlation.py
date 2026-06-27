import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_PASSES_FILE = Path("vehicle_day_zone_passes.csv")


def _pearson(left: list[float], right: list[float]) -> float | None:
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


def _load_vehicle_passes(path: Path, include_any_zone: bool) -> tuple[dict[int, dict[str, tuple[int, ...]]], list[str]]:
    vehicles: dict[int, dict[str, tuple[int, ...]]] = defaultdict(dict)

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")

        zone_columns = [column for column in reader.fieldnames if column.startswith("passes_zone_")]
        if include_any_zone:
            zone_columns = ["passes_any_zone", *zone_columns]

        if not zone_columns:
            raise ValueError(f"{path} does not contain pass indicator columns.")

        for row in reader:
            vehicle_id = int(row["VehicleId"])
            date = row["date"]
            vehicles[vehicle_id][date] = tuple(int(row[column]) for column in zone_columns)

    return dict(vehicles), zone_columns


def _flatten_common_dates(
    left: dict[str, tuple[int, ...]],
    right: dict[str, tuple[int, ...]],
) -> tuple[list[float], list[float], int]:
    common_dates = sorted(set(left) & set(right))
    left_values: list[float] = []
    right_values: list[float] = []

    for date in common_dates:
        left_values.extend(left[date])
        right_values.extend(right[date])

    return left_values, right_values, len(common_dates)


def calculate_truck_correlations(
    passes_file: Path,
    target_vehicle: int,
    include_any_zone: bool = False,
) -> tuple[list[dict[str, str]], list[str]]:
    vehicles, zone_columns = _load_vehicle_passes(passes_file, include_any_zone)
    if target_vehicle not in vehicles:
        available = ", ".join(str(vehicle_id) for vehicle_id in sorted(vehicles)[:10])
        raise ValueError(f"Vehicle {target_vehicle} was not found. First available vehicles: {available}")

    rows = []
    target_passes = vehicles[target_vehicle]
    for vehicle_id, passes in vehicles.items():
        if vehicle_id == target_vehicle:
            continue

        target_values, other_values, common_date_count = _flatten_common_dates(target_passes, passes)
        correlation = _pearson(target_values, other_values)
        if correlation is None:
            continue

        rows.append(
            {
                "target_vehicle": str(target_vehicle),
                "vehicle_id": str(vehicle_id),
                "correlation": f"{correlation:.6f}",
                "common_dates": str(common_date_count),
                "features_compared": str(len(target_values)),
            }
        )

    rows.sort(key=lambda row: float(row["correlation"]), reverse=True)
    return rows, zone_columns


def write_correlations(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["target_vehicle", "vehicle_id", "correlation", "common_dates", "features_compared"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate truck-to-truck correlation from daily zone pass indicators."
    )
    parser.add_argument("--passes-file", type=Path, default=DEFAULT_PASSES_FILE)
    parser.add_argument("--target-vehicle", type=int, default=1994)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--include-any-zone", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows, zone_columns = calculate_truck_correlations(
        args.passes_file,
        args.target_vehicle,
        include_any_zone=args.include_any_zone,
    )

    output_path = args.output or Path(f"truck_correlations_{args.target_vehicle}.csv")
    write_correlations(output_path, rows)

    print(f"Compared Vehicle {args.target_vehicle} using {', '.join(zone_columns)}.")
    print(f"Wrote {len(rows)} correlations to {output_path}.")
    for row in rows[: args.top]:
        print(
            f"Vehicle {row['vehicle_id']}: correlation={row['correlation']}, "
            f"common_dates={row['common_dates']}"
        )


if __name__ == "__main__":
    main()
