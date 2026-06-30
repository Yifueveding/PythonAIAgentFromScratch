import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_PASSES_FILE = Path("vehicle_day_zone_passes.csv")
DEFAULT_OUTPUT_DIR = Path("correlation_calculator")


@dataclass(frozen=True)
class StopProfile:
    vehicle_id: int
    start_date: str
    end_date: str
    observed_days: int
    vector_labels: list[str]
    vector: list[float]


def _normalize_date(value: str) -> str:
    for date_format in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value[:10], date_format).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value[:10]


def _date_kind(date: str) -> str:
    day = datetime.strptime(date, "%Y-%m-%d")
    return "weekend" if day.weekday() >= 5 else "weekday"


def _in_period(date: str, start_date: str, end_date: str) -> bool:
    return start_date <= date <= end_date


def _pearson(left: list[float], right: list[float]) -> Optional[float]:
    if len(left) != len(right) or len(left) < 2:
        return None

    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_denominator = sum((a - left_mean) ** 2 for a in left)
    right_denominator = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_denominator * right_denominator)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _load_stop_profiles(
    passes_file: Path,
    vehicle_ids: list[int],
    start_date: str,
    end_date: str,
) -> tuple[dict[int, StopProfile], list[str]]:
    selected_vehicles = set(vehicle_ids)
    profile_values: dict[int, dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
    observed_days: dict[int, set[str]] = defaultdict(set)

    with passes_file.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{passes_file} has no header row.")

        zone_columns = [column for column in reader.fieldnames if column.startswith("passes_zone_")]
        if not zone_columns:
            raise ValueError(f"{passes_file} does not contain passes_zone_* columns.")

        for row in reader:
            vehicle_id = int(row["VehicleId"])
            if vehicle_id not in selected_vehicles:
                continue

            date = _normalize_date(row["date"])
            if not _in_period(date, start_date, end_date):
                continue

            day_kind = _date_kind(date)
            observed_days[vehicle_id].add(date)
            for zone in zone_columns:
                profile_values[vehicle_id][(zone, day_kind)] += float(row[zone])

    vector_labels = [
        f"{zone}_{day_kind}"
        for day_kind in ("weekday", "weekend")
        for zone in zone_columns
    ]
    profiles = {}
    for vehicle_id in vehicle_ids:
        values = profile_values[vehicle_id]
        profiles[vehicle_id] = StopProfile(
            vehicle_id=vehicle_id,
            start_date=start_date,
            end_date=end_date,
            observed_days=len(observed_days[vehicle_id]),
            vector_labels=vector_labels,
            vector=[
                values[(zone, day_kind)]
                for day_kind in ("weekday", "weekend")
                for zone in zone_columns
            ],
        )

    return profiles, zone_columns


def calculate_period_correlations(
    passes_file: Path,
    vehicle_ids: list[int],
    start_date: str,
    end_date: str,
) -> dict:
    normalized_start = _normalize_date(start_date)
    normalized_end = _normalize_date(end_date)
    if normalized_start > normalized_end:
        raise ValueError("--start-date must be before or equal to --end-date.")

    profiles, zone_columns = _load_stop_profiles(
        passes_file,
        vehicle_ids,
        normalized_start,
        normalized_end,
    )

    rows = []
    for index, vehicle_a in enumerate(vehicle_ids):
        for vehicle_b in vehicle_ids[index + 1 :]:
            profile_a = profiles[vehicle_a]
            profile_b = profiles[vehicle_b]
            correlation = _pearson(profile_a.vector, profile_b.vector)
            rows.append(
                {
                    "vehicle_a": vehicle_a,
                    "vehicle_b": vehicle_b,
                    "start_date": normalized_start,
                    "end_date": normalized_end,
                    "correlation": correlation,
                    "features_compared": len(profile_a.vector),
                    "vehicle_a_observed_days": profile_a.observed_days,
                    "vehicle_b_observed_days": profile_b.observed_days,
                }
            )

    rows.sort(
        key=lambda row: row["correlation"] if row["correlation"] is not None else -2.0,
        reverse=True,
    )

    return {
        "method": {
            "description": (
                "For each selected truck, build one stop-profile vector across the selected period. "
                "The vector has dimensions passes_zone_* x weekday/weekend, and each element is "
                "the sum of passing counts over that period."
            ),
            "period": {"start_date": normalized_start, "end_date": normalized_end},
            "zone_columns": zone_columns,
            "vector_order": next(iter(profiles.values())).vector_labels if profiles else [],
        },
        "profiles": {
            str(vehicle_id): asdict(profile)
            for vehicle_id, profile in profiles.items()
        },
        "pairwise_correlations": rows,
    }


def _write_correlations(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "vehicle_a",
        "vehicle_b",
        "start_date",
        "end_date",
        "correlation",
        "features_compared",
        "vehicle_a_observed_days",
        "vehicle_b_observed_days",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = dict(row)
            if output_row["correlation"] is not None:
                output_row["correlation"] = f"{output_row['correlation']:.6f}"
            writer.writerow(output_row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate truck correlations over a selected period using stop-profile "
            "vectors with dimensions passes_zone_* x weekday/weekend."
        )
    )
    parser.add_argument("vehicles", nargs="+", type=int, help="Vehicle IDs to compare.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--passes-file", type=Path, default=DEFAULT_PASSES_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    result = calculate_period_correlations(
        args.passes_file,
        args.vehicles,
        args.start_date,
        args.end_date,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vehicle_label = "_".join(str(vehicle_id) for vehicle_id in args.vehicles)
    start_date = result["method"]["period"]["start_date"]
    end_date = result["method"]["period"]["end_date"]
    json_file = output_dir / f"period_{start_date}_{end_date}_vehicles_{vehicle_label}.json"
    csv_file = output_dir / f"period_{start_date}_{end_date}_vehicles_{vehicle_label}.csv"
    profiles_file = output_dir / f"period_{start_date}_{end_date}_vehicles_{vehicle_label}_stop_profiles.json"
    json_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    profiles_file.write_text(json.dumps(result["profiles"], indent=2) + "\n", encoding="utf-8")
    _write_correlations(csv_file, result["pairwise_correlations"])

    print(f"Wrote full result to {json_file}")
    print(f"Wrote stop-profile vectors to {profiles_file}")
    print(f"Wrote pairwise correlations to {csv_file}")
    print("Top correlations:")
    for row in result["pairwise_correlations"][: args.top]:
        correlation = row["correlation"]
        correlation_text = "n/a" if correlation is None else f"{correlation:.3f}"
        print(
            f"  Vehicle {row['vehicle_a']} + {row['vehicle_b']}: "
            f"correlation={correlation_text}, "
            f"observed_days=({row['vehicle_a_observed_days']}, {row['vehicle_b_observed_days']})"
        )


if __name__ == "__main__":
    main()
