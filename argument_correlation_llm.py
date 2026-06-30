import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from argument_correlation import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WEIGHTS,
    _parse_weight_overrides,
    calculate_statistical_argument_correlation,
)
from correlation_calculator import DEFAULT_PASSES_FILE
from data_representation import DEFAULT_FLEET_FILE, DEFAULT_PURPOSE_FILE
from image_cluster import DEFAULT_CLUSTERS, DEFAULT_IMAGE_ROOT, DEFAULT_IMAGE_SIZE


DEFAULT_LLM_OUTPUT_DIR = Path("argument_correlation_llm")
DEFAULT_DISTANCE_FILE = Path("vehicle_daily_distance_speed.csv")


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


def _build_llm(provider: str = "anthropic", model_name: Optional[str] = None):
    load_dotenv(".env", override=False)
    load_dotenv("sample.env", override=False)
    load_dotenv("sample1.env", override=True)
    for env_key in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        if os.getenv(env_key):
            os.environ[env_key] = os.environ[env_key].strip()

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise RuntimeError("langchain-anthropic is required for argument_correlation_llm.py.") from e
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for --llm-provider anthropic.")
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


def _bounded_score(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _summarize_representative_routes(result: dict) -> dict:
    return {
        vehicle_id: [
            {
                "cluster": route.get("cluster"),
                "representative_date": route.get("representative_date"),
                "share": route.get("share"),
            }
            for route in routes
        ]
        for vehicle_id, routes in result.get("representative_routes", {}).items()
    }


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _stdev(values: list[float]) -> Optional[float]:
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None


def _as_float(value: object) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _in_period(date: str, start_date: str, end_date: str) -> bool:
    return start_date <= date[:10] <= end_date


def _load_distance_summary(
    distance_file: Path,
    vehicle_ids: list[int],
    start_date: str,
    end_date: str,
) -> dict[str, dict]:
    wanted = {str(vehicle_id) for vehicle_id in vehicle_ids}
    distances: dict[str, list[float]] = {vehicle_id: [] for vehicle_id in wanted}
    speeds: dict[str, list[float]] = {vehicle_id: [] for vehicle_id in wanted}
    active_dates: dict[str, list[str]] = {vehicle_id: [] for vehicle_id in wanted}

    if not distance_file.exists():
        return {
            vehicle_id: {"available": False, "reason": f"missing {distance_file}"}
            for vehicle_id in wanted
        }

    with distance_file.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            vehicle_id = str(row.get("VehicleId", ""))
            date = str(row.get("date", ""))[:10]
            if vehicle_id not in wanted or not _in_period(date, start_date, end_date):
                continue
            distance = _as_float(row.get("total_distance_km"))
            speed = _as_float(row.get("average_speed"))
            if distance is not None:
                distances[vehicle_id].append(distance)
                active_dates[vehicle_id].append(date)
            if speed is not None:
                speeds[vehicle_id].append(speed)

    summaries = {}
    for vehicle_id in wanted:
        vehicle_distances = distances[vehicle_id]
        summaries[vehicle_id] = {
            "available": bool(vehicle_distances),
            "active_days": len(set(active_dates[vehicle_id])),
            "distance_km_mean": _mean(vehicle_distances),
            "distance_km_median": statistics.median(vehicle_distances) if vehicle_distances else None,
            "distance_km_stdev": _stdev(vehicle_distances),
            "distance_km_min": min(vehicle_distances) if vehicle_distances else None,
            "distance_km_max": max(vehicle_distances) if vehicle_distances else None,
            "average_speed_mean": _mean(speeds[vehicle_id]),
            "sample_dates": sorted(set(active_dates[vehicle_id]))[:8],
        }
    return summaries


def _load_zone_summary(
    passes_file: Path,
    vehicle_ids: list[int],
    start_date: str,
    end_date: str,
) -> dict[str, dict]:
    wanted = {str(vehicle_id) for vehicle_id in vehicle_ids}
    zone_columns = [f"passes_zone_{index}" for index in range(1, 9)]
    summaries = {
        vehicle_id: {
            "observed_days": 0,
            "passes_any_zone_total": 0.0,
            "zone_totals": {column: 0.0 for column in zone_columns},
            "weekday_total": 0.0,
            "weekend_total": 0.0,
        }
        for vehicle_id in wanted
    }
    if not passes_file.exists():
        return {
            vehicle_id: {"available": False, "reason": f"missing {passes_file}"}
            for vehicle_id in wanted
        }

    from datetime import datetime

    with passes_file.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            vehicle_id = str(row.get("VehicleId", ""))
            date = str(row.get("date", ""))[:10]
            if vehicle_id not in wanted or not _in_period(date, start_date, end_date):
                continue
            summary = summaries[vehicle_id]
            summary["observed_days"] += 1
            any_zone = _as_float(row.get("passes_any_zone")) or 0.0
            summary["passes_any_zone_total"] += any_zone
            try:
                is_weekend = datetime.strptime(date, "%Y-%m-%d").weekday() >= 5
            except ValueError:
                is_weekend = False
            if is_weekend:
                summary["weekend_total"] += any_zone
            else:
                summary["weekday_total"] += any_zone
            for column in zone_columns:
                summary["zone_totals"][column] += _as_float(row.get(column)) or 0.0

    for summary in summaries.values():
        totals = summary["zone_totals"]
        summary["dominant_zones"] = [
            {"zone": zone, "passes": total}
            for zone, total in sorted(totals.items(), key=lambda item: item[1], reverse=True)
            if total > 0
        ][:3]
    return summaries


def _load_total_distance_summary(total_distance_file: Path, vehicle_ids: list[int]) -> dict[str, dict]:
    wanted = {str(vehicle_id) for vehicle_id in vehicle_ids}
    if not total_distance_file.exists():
        return {}
    summaries = {}
    with total_distance_file.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            vehicle_id = str(row.get("VehicleId", row.get("vehicle_id", "")))
            if vehicle_id not in wanted:
                continue
            summaries[vehicle_id] = dict(row)
    return summaries


def _cluster_distance_summary(
    image_cluster_results: Optional[dict],
    distance_file: Path,
    vehicle_ids: list[int],
    start_date: str,
    end_date: str,
) -> dict[str, list[dict]]:
    if not image_cluster_results or not distance_file.exists():
        return {}

    distances_by_vehicle_date: dict[tuple[str, str], float] = {}
    wanted = {str(vehicle_id) for vehicle_id in vehicle_ids}
    with distance_file.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            vehicle_id = str(row.get("VehicleId", ""))
            date = str(row.get("date", ""))[:10]
            if vehicle_id not in wanted or not _in_period(date, start_date, end_date):
                continue
            distance = _as_float(row.get("total_distance_km"))
            if distance is not None:
                distances_by_vehicle_date[(vehicle_id, date)] = distance

    output = {}
    for vehicle_id in wanted:
        cluster_result = image_cluster_results.get(vehicle_id) or image_cluster_results.get(int(vehicle_id))
        if not cluster_result:
            continue
        distances_by_cluster: dict[int, list[float]] = defaultdict(list)
        for row in cluster_result.get("clusters_by_date", []):
            date = row.get("date")
            cluster = row.get("cluster")
            distance = distances_by_vehicle_date.get((vehicle_id, date))
            if cluster is not None and distance is not None:
                distances_by_cluster[int(cluster)].append(distance)
        output[vehicle_id] = [
            {
                "cluster": cluster,
                "distance_km_mean": _mean(values),
                "distance_km_median": statistics.median(values) if values else None,
                "distance_km_min": min(values) if values else None,
                "distance_km_max": max(values) if values else None,
                "distance_sample_count": len(values),
            }
            for cluster, values in sorted(distances_by_cluster.items())
        ]
    return output


def _build_operational_evidence(
    vehicle_ids: list[int],
    start_date: str,
    end_date: str,
    passes_file: Path,
    distance_file: Path,
    image_cluster_results: Optional[dict],
) -> dict:
    return {
        "source_files": {
            "fleet_metadata": str(DEFAULT_FLEET_FILE),
            "truck_purpose": str(DEFAULT_PURPOSE_FILE),
            "daily_distance_speed": str(distance_file),
            "zone_passes": str(passes_file),
            "total_distance": "Other_data/Total_Distance.csv",
        },
        "distance_speed_summary": _load_distance_summary(distance_file, vehicle_ids, start_date, end_date),
        "zone_pass_summary": _load_zone_summary(passes_file, vehicle_ids, start_date, end_date),
        "total_distance_summary": _load_total_distance_summary(Path("Other_data/Total_Distance.csv"), vehicle_ids),
        "route_cluster_distance_summary": _cluster_distance_summary(
            image_cluster_results,
            distance_file,
            vehicle_ids,
            start_date,
            end_date,
        ),
    }


def _llm_payload_from_statistical_result(result: dict, operational_evidence: Optional[dict] = None) -> dict:
    return {
        "method": result["method"],
        "fleet_metadata": result["fleet_metadata"],
        "representative_routes": _summarize_representative_routes(result),
        "operational_evidence": operational_evidence or {},
        "pairwise_evidence": [
            {
                "vehicle_a": row["vehicle_a"],
                "vehicle_b": row["vehicle_b"],
                "statistical_total_correlation": row["total_correlation"],
                "features": row["features"],
                "weights": row["weights"],
                "vehicle_a_observed_days": row["vehicle_a_observed_days"],
                "vehicle_b_observed_days": row["vehicle_b_observed_days"],
            }
            for row in result["pairwise_argument_correlations"]
        ],
    }


def _request_llm_correlations(
    statistical_result: dict,
    provider: str,
    model_name: Optional[str],
    operational_evidence: Optional[dict] = None,
) -> dict:
    llm, resolved_model = _build_llm(provider, model_name)
    payload = _llm_payload_from_statistical_result(statistical_result, operational_evidence)
    prompt = f"""
You are reasoning about truck-to-truck behavioral similarity for scenario generation.

Task:
- Review the pairwise evidence for each truck pair.
- Produce a reasoned total_correlation from 0.0 to 1.0 for each pair.
- Use the statistical score as evidence, but you may adjust it when the feature pattern suggests the linear weighted score overstates or understates similarity.
- Consider stop-profile correlation, representative route-image similarity, truck purpose, age similarity, kilometer similarity, duty-cycle match, observed days, route-cluster shares, daily distance/speed statistics, zone-pass summaries, and route-cluster distance summaries.
- Use distance/speed and zone-pass summaries to identify trucks with similar operating intensity, route length, and stop-zone behavior.
- Do not invent features or use external knowledge.

Rules:
- Return only one JSON object.
- Return every pair exactly once.
- total_correlation must be between 0.0 and 1.0.
- Keep rationales concise.

Input evidence:
{json.dumps(payload, indent=2)}

Return this schema:
{{
  "pairwise_argument_correlations": [
    {{
      "vehicle_a": 155,
      "vehicle_b": 1181,
      "total_correlation": 0.0,
      "rationale": "short reason",
      "feature_interpretation": {{
        "stop_profile_correlation": "short reason",
        "route_similarity": "short reason",
        "fleet_metadata": "short reason"
      }}
    }}
  ],
  "global_rationale": "short explanation of how similarities were reasoned",
  "risks": ["short risk"]
}}
"""
    response = llm.invoke(prompt)
    parsed = _extract_json_object(_llm_json_text(response))
    parsed["llm_provider"] = provider
    parsed["llm_model"] = resolved_model
    return parsed


def _merge_llm_correlations(statistical_result: dict, llm_result: dict) -> list[dict]:
    llm_by_pair = {}
    for row in llm_result.get("pairwise_argument_correlations", []):
        try:
            key = tuple(sorted((int(row["vehicle_a"]), int(row["vehicle_b"]))))
        except (KeyError, TypeError, ValueError):
            continue
        llm_by_pair[key] = row

    merged_rows = []
    for row in statistical_result["pairwise_argument_correlations"]:
        key = tuple(sorted((int(row["vehicle_a"]), int(row["vehicle_b"]))))
        llm_row = llm_by_pair.get(key, {})
        llm_score = _bounded_score(llm_row.get("total_correlation"))
        statistical_score = row["total_correlation"]
        total_correlation = llm_score if llm_score is not None else statistical_score
        merged_rows.append(
            {
                **row,
                "method": "llm_reasoned",
                "total_correlation": total_correlation,
                "statistical_total_correlation": statistical_score,
                "llm_rationale": llm_row.get("rationale", ""),
                "llm_feature_interpretation": llm_row.get("feature_interpretation", {}),
            }
        )

    merged_rows.sort(
        key=lambda row: row["total_correlation"] if row["total_correlation"] is not None else -2.0,
        reverse=True,
    )
    return merged_rows


def calculate_llm_argument_correlation(
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
    distance_file: Path = DEFAULT_DISTANCE_FILE,
    llm_provider: str = "anthropic",
    llm_model: Optional[str] = None,
) -> dict:
    statistical_result = calculate_statistical_argument_correlation(
        vehicle_ids=vehicle_ids,
        start_date=start_date,
        end_date=end_date,
        passes_file=passes_file,
        purpose_file=purpose_file,
        fleet_file=fleet_file,
        weights=weights,
        image_cluster_results=image_cluster_results,
        image_root=image_root,
        route_cluster_output_dir=route_cluster_output_dir,
        route_feature_method=route_feature_method,
        route_clusters=route_clusters,
        route_image_size=route_image_size,
    )
    normalized_start = statistical_result["method"]["period"]["start_date"]
    normalized_end = statistical_result["method"]["period"]["end_date"]
    operational_evidence = _build_operational_evidence(
        vehicle_ids=vehicle_ids,
        start_date=normalized_start,
        end_date=normalized_end,
        passes_file=passes_file,
        distance_file=distance_file,
        image_cluster_results=image_cluster_results,
    )
    llm_result = _request_llm_correlations(
        statistical_result,
        llm_provider,
        llm_model,
        operational_evidence=operational_evidence,
    )
    rows = _merge_llm_correlations(statistical_result, llm_result)
    return {
        **statistical_result,
        "method": {
            **statistical_result["method"],
            "name": "llm_reasoned",
            "description": (
                "Compute statistical pairwise evidence, then ask an LLM to reason over "
                "stop-profile, representative route-image, and fleet-metadata evidence "
                "to produce final pairwise total_correlation values."
            ),
            "llm_provider": llm_result["llm_provider"],
            "llm_model": llm_result["llm_model"],
            "global_rationale": llm_result.get("global_rationale", ""),
            "risks": llm_result.get("risks", []),
        },
        "operational_evidence": operational_evidence,
        "statistical_pairwise_argument_correlations": statistical_result["pairwise_argument_correlations"],
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
        "statistical_total_correlation",
        "stop_profile_correlation",
        "route_similarity",
        "purpose_match",
        "age_similarity",
        "kms_similarity",
        "duty_cycle_match",
        "llm_rationale",
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
                "statistical_total_correlation": row["statistical_total_correlation"],
                "llm_rationale": row.get("llm_rationale", ""),
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
        description="Calculate LLM-reasoned argument correlations from statistical truck-pair evidence."
    )
    parser.add_argument("vehicles", nargs="+", type=int, help="Vehicle IDs to compare.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--passes-file", type=Path, default=DEFAULT_PASSES_FILE)
    parser.add_argument("--purpose-file", type=Path, default=DEFAULT_PURPOSE_FILE)
    parser.add_argument("--fleet-file", type=Path, default=DEFAULT_FLEET_FILE)
    parser.add_argument("--distance-file", type=Path, default=DEFAULT_DISTANCE_FILE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--route-cluster-output-dir", type=Path, default=DEFAULT_LLM_OUTPUT_DIR / "route_clusters")
    parser.add_argument("--route-feature-method", choices=("pca", "autoencoder"), default="pca")
    parser.add_argument("--route-clusters", type=int, default=DEFAULT_CLUSTERS)
    parser.add_argument("--route-image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LLM_OUTPUT_DIR)
    parser.add_argument("--llm-provider", choices=("anthropic", "openai"), default="anthropic")
    parser.add_argument("--llm-model")
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help="Override a statistical evidence weight as key=value. Allowed keys: " + ", ".join(DEFAULT_WEIGHTS),
    )
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    result = calculate_llm_argument_correlation(
        vehicle_ids=args.vehicles,
        start_date=args.start_date,
        end_date=args.end_date,
        passes_file=args.passes_file,
        purpose_file=args.purpose_file,
        fleet_file=args.fleet_file,
        distance_file=args.distance_file,
        weights=_parse_weight_overrides(args.weight),
        image_root=args.image_root,
        route_cluster_output_dir=args.route_cluster_output_dir,
        route_feature_method=args.route_feature_method,
        route_clusters=args.route_clusters,
        route_image_size=args.route_image_size,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_date = result["method"]["period"]["start_date"]
    end_date = result["method"]["period"]["end_date"]
    vehicle_label = "_".join(str(vehicle_id) for vehicle_id in args.vehicles)
    base_name = f"llm_period_{start_date}_{end_date}_vehicles_{vehicle_label}"
    json_file = args.output_dir / f"{base_name}.json"
    csv_file = args.output_dir / f"{base_name}.csv"
    json_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_file, result["pairwise_argument_correlations"])

    print(f"Wrote LLM argument correlations to {json_file}")
    print(f"Wrote pairwise CSV to {csv_file}")
    for row in result["pairwise_argument_correlations"][: args.top]:
        score = row["total_correlation"]
        stat_score = row["statistical_total_correlation"]
        print(
            f"  Vehicle {row['vehicle_a']} + {row['vehicle_b']}: "
            f"llm_total={score:.3f}, statistical_total={stat_score:.3f}"
        )


if __name__ == "__main__":
    main()
