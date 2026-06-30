import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from math import nan
from pathlib import Path
from typing import Optional


DEFAULT_STOP_ZONE_FILE = Path("vehicle_day_zone_passes.csv")
DEFAULT_DISTANCE_FILE = Path("vehicle_daily_distance_speed.csv")
DEFAULT_PURPOSE_FILE = Path("Other_data/truck_purpose.csv")
DEFAULT_FLEET_FILE = Path("Other_data/West_Ang_Fleet.csv")
DEFAULT_IMAGE_ROOT = Path("Vehicle")
DEFAULT_OUTPUT_DIR = Path("data_representation")


@dataclass(frozen=True)
class StopZoneRecord:
    passes_any_zone: Optional[int]
    zones: dict[str, int]


@dataclass(frozen=True)
class RouteDistanceRecord:
    total_distance_miles: Optional[float]
    total_distance_km: Optional[float]
    average_speed: Optional[float]
    point_count: Optional[int]
    moving_point_count: Optional[int]
    kept_segments: Optional[int]
    dropped_segments: Optional[int]


@dataclass(frozen=True)
class FleetMetadata:
    purpose: Optional[str]
    age: Optional[float]
    kms: Optional[float]
    duty_cycle: Optional[str]


@dataclass(frozen=True)
class TruckDayRepresentation:
    vehicle_id: int
    date: str
    stop_zone_record: Optional[StopZoneRecord]
    gps_image: Optional[str]
    route_distance: Optional[RouteDistanceRecord]
    fleet_metadata: FleetMetadata


def _parse_float(value: object) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _parse_int(value: object) -> Optional[int]:
    if value in ("", None):
        return None
    try:
        return int(float(str(value).replace(",", "")))
    except ValueError:
        return None


def _normalize_date(value: str) -> str:
    for date_format in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value[:10], date_format).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value[:10]


def _replace_none_with_nan(value: object) -> object:
    if value is None:
        return nan
    if isinstance(value, dict):
        return {key: _replace_none_with_nan(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_none_with_nan(item) for item in value]
    return value


def _vehicle_image_path(image_root: Path, vehicle_id: int, date: str) -> Optional[Path]:
    nested_path = image_root / f"Vehicle_{vehicle_id}" / f"{date}.png"
    if nested_path.exists():
        return nested_path

    flat_path = Path(f"Vehicle_{vehicle_id}") / f"{date}.png"
    if flat_path.exists():
        return flat_path

    return None


def _load_stop_zone_records(path: Path) -> tuple[dict[tuple[int, str], StopZoneRecord], list[str]]:
    records = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return records, []

        zone_columns = [column for column in reader.fieldnames if column.startswith("passes_zone_")]
        for row in reader:
            vehicle_id = _parse_int(row.get("VehicleId") or row.get("vehicle_id"))
            date = row.get("date") or row.get("Date")
            if vehicle_id is None or not date:
                continue

            records[(vehicle_id, date[:10])] = StopZoneRecord(
                passes_any_zone=_parse_int(row.get("passes_any_zone")),
                zones={column: _parse_int(row.get(column)) or 0 for column in zone_columns},
            )
    return records, zone_columns


def _zero_stop_zone_record(zone_columns: list[str]) -> StopZoneRecord:
    return StopZoneRecord(
        passes_any_zone=0,
        zones={column: 0 for column in zone_columns},
    )


def _load_route_distances(path: Path) -> dict[tuple[int, str], RouteDistanceRecord]:
    records = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return records

        for row in reader:
            vehicle_id = _parse_int(row.get("VehicleId") or row.get("vehicle_id"))
            date = row.get("date") or row.get("Date")
            if vehicle_id is None or not date:
                continue

            records[(vehicle_id, date[:10])] = RouteDistanceRecord(
                total_distance_miles=_parse_float(row.get("total_distance_miles")),
                total_distance_km=_parse_float(row.get("total_distance_km")),
                average_speed=_parse_float(row.get("average_speed")),
                point_count=_parse_int(row.get("point_count")),
                moving_point_count=_parse_int(row.get("moving_point_count")),
                kept_segments=_parse_int(row.get("kept_segments")),
                dropped_segments=_parse_int(row.get("dropped_segments")),
            )
    return records


def _load_purpose(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}

    purposes = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return purposes

        vehicle_column = "VehicleId" if "VehicleId" in reader.fieldnames else "vehicle_id"
        purpose_column = "purpose" if "purpose" in reader.fieldnames else "truck_purpose"
        for row in reader:
            vehicle_id = _parse_int(row.get(vehicle_column))
            purpose = row.get(purpose_column)
            if vehicle_id is not None and purpose:
                purposes[vehicle_id] = purpose
    return purposes


def _load_fleet_metadata(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    if not rows:
        return {}

    header = rows[0]

    def first_index(name: str) -> Optional[int]:
        for index, value in enumerate(header):
            if value == name:
                return index
        return None

    def all_indexes(name: str) -> list[int]:
        return [index for index, value in enumerate(header) if value == name]

    vehicle_index = first_index("VehicleId")
    age_index = first_index("Age")
    kms_index = first_index("Km's")
    duty_cycle_indexes = all_indexes("Duty Cycle")

    if vehicle_index is None:
        return {}

    metadata = {}
    for row in rows[1:]:
        if len(row) <= vehicle_index:
            continue
        vehicle_id = _parse_int(row[vehicle_index])
        if vehicle_id is None:
            continue

        duty_cycle = None
        for index in duty_cycle_indexes:
            if len(row) > index and row[index].strip():
                duty_cycle = row[index].strip()
                break

        metadata[vehicle_id] = {
            "age": _parse_float(row[age_index]) if age_index is not None and len(row) > age_index else None,
            "kms": _parse_float(row[kms_index]) if kms_index is not None and len(row) > kms_index else None,
            "duty_cycle": duty_cycle,
        }
    return metadata


class TruckDayLookup:
    """Lookup for a truck-day multi-modal representation."""

    def __init__(
        self,
        stop_zone_file: Path = DEFAULT_STOP_ZONE_FILE,
        distance_file: Path = DEFAULT_DISTANCE_FILE,
        purpose_file: Path = DEFAULT_PURPOSE_FILE,
        fleet_file: Path = DEFAULT_FLEET_FILE,
        image_root: Path = DEFAULT_IMAGE_ROOT,
    ) -> None:
        self.image_root = image_root
        self.stop_zone_records, self.zone_columns = _load_stop_zone_records(stop_zone_file)
        self.route_distances = _load_route_distances(distance_file)
        self.purposes = _load_purpose(purpose_file)
        self.fleet_metadata = _load_fleet_metadata(fleet_file)

    def get(self, vehicle_id: int, date: str) -> TruckDayRepresentation:
        normalized_date = _normalize_date(date)
        fleet = self.fleet_metadata.get(vehicle_id, {})
        image_path = _vehicle_image_path(self.image_root, vehicle_id, normalized_date)
        route_distance = self.route_distances.get((vehicle_id, normalized_date))
        stop_zone_record = self.stop_zone_records.get((vehicle_id, normalized_date))
        if stop_zone_record is None and image_path is not None:
            stop_zone_record = _zero_stop_zone_record(self.zone_columns)

        return TruckDayRepresentation(
            vehicle_id=vehicle_id,
            date=normalized_date,
            stop_zone_record=stop_zone_record,
            gps_image=str(image_path) if image_path is not None else None,
            route_distance=route_distance,
            fleet_metadata=FleetMetadata(
                purpose=self.purposes.get(vehicle_id),
                age=fleet.get("age"),
                kms=fleet.get("kms"),
                duty_cycle=fleet.get("duty_cycle"),
            ),
        )

    def available_dates(self, vehicle_id: int) -> list[str]:
        dates = {
            date
            for current_vehicle_id, date in self.stop_zone_records
            if current_vehicle_id == vehicle_id
        }
        dates.update(
            date
            for current_vehicle_id, date in self.route_distances
            if current_vehicle_id == vehicle_id
        )
        image_dir = self.image_root / f"Vehicle_{vehicle_id}"
        if image_dir.exists():
            dates.update(path.stem for path in image_dir.glob("*.png"))
        return sorted(dates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Represent one truck and day with stop-zone, GPS-image, route-distance, and fleet metadata."
    )
    parser.add_argument("--vehicle-id", type=int, required=True)
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD format.")
    parser.add_argument("--stop-zone-file", type=Path, default=DEFAULT_STOP_ZONE_FILE)
    parser.add_argument("--distance-file", type=Path, default=DEFAULT_DISTANCE_FILE)
    parser.add_argument("--purpose-file", type=Path, default=DEFAULT_PURPOSE_FILE)
    parser.add_argument("--fleet-file", type=Path, default=DEFAULT_FLEET_FILE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    lookup = TruckDayLookup(
        stop_zone_file=args.stop_zone_file,
        distance_file=args.distance_file,
        purpose_file=args.purpose_file,
        fleet_file=args.fleet_file,
        image_root=args.image_root,
    )
    representation = lookup.get(args.vehicle_id, args.date)
    result = _replace_none_with_nan(asdict(representation))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"vehicle_{representation.vehicle_id}_{representation.date}.json"
    output_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Saved representation to {output_file}")


if __name__ == "__main__":
    main()
