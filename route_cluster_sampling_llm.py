import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from argument_correlation_llm import (
    DEFAULT_DISTANCE_FILE,
    _build_llm,
    _invoke_json,
)
from main import _normalize_date


DEFAULT_OUTPUT_DIR = Path("route_cluster_sampling_llm")


def _as_float(value: object) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(path_text: str, scenario_summary_path: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidate = scenario_summary_path.parent / path.name
    if candidate.exists():
        return candidate
    return path


def _load_distance_by_vehicle_date(
    distance_file: Path,
    vehicle_ids: list[int],
    start_date: str,
    end_date: str,
) -> dict[tuple[int, str], float]:
    wanted = {str(vehicle_id) for vehicle_id in vehicle_ids}
    distances = {}
    with distance_file.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            vehicle_id = str(row.get("VehicleId", ""))
            date = str(row.get("date", ""))[:10]
            if vehicle_id not in wanted or not (start_date <= date <= end_date):
                continue
            distance = _as_float(row.get("total_distance_km"))
            if distance is not None:
                distances[(int(vehicle_id), date)] = distance
    return distances


def _distance_stats(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean_km": None,
            "min_km": None,
            "max_km": None,
        }
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        median = sorted_values[midpoint]
    else:
        median = (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2
    return {
        "count": len(values),
        "mean_km": sum(values) / len(values),
        "median_km": median,
        "min_km": min(values),
        "max_km": max(values),
    }


def _cluster_distance_stats(
    image_clusters: dict,
    vehicle_ids: list[int],
    distances: dict[tuple[int, str], float],
) -> dict[str, dict[int, dict]]:
    stats = {}
    for vehicle_id in vehicle_ids:
        cluster_distances = defaultdict(list)
        cluster_result = image_clusters.get(str(vehicle_id), {})
        for row in cluster_result.get("clusters_by_date", []):
            date = row.get("date")
            cluster = row.get("cluster")
            distance = distances.get((vehicle_id, date))
            if cluster is not None and distance is not None:
                cluster_distances[int(cluster)].append(distance)
        stats[str(vehicle_id)] = {
            cluster: _distance_stats(values)
            for cluster, values in cluster_distances.items()
        }
    return stats


def _build_cluster_options(
    scenario_summary: dict,
    image_clusters: dict,
    distance_stats: dict[str, dict[int, dict]],
) -> dict[str, list[dict]]:
    options = {}
    target_vehicle = int(scenario_summary["inputs"]["target_vehicle"])
    target_rows = scenario_summary.get("target_route_appearance_frequencies", [])
    options[str(target_vehicle)] = [
        {
            "vehicle_id": target_vehicle,
            "cluster": row["cluster"],
            "current_route_weight": row.get("route_appearance_frequency"),
            "route_days": row.get("route_days"),
            "representative_date": row.get("representative_date"),
            "example_dates": (row.get("example_dates") or [])[:8],
            "distance_stats": distance_stats.get(str(target_vehicle), {}).get(int(row["cluster"]), {}),
            "role": "target_vehicle",
        }
        for row in target_rows
    ]

    for row in scenario_summary.get("route_appearance_frequencies", []):
        vehicle_id = int(row["vehicle_id"])
        options.setdefault(str(vehicle_id), []).append(
            {
                "vehicle_id": vehicle_id,
                "cluster": row["cluster"],
                "current_route_weight": row.get("route_appearance_frequency"),
                "truck_appearance_frequency": row.get("truck_appearance_frequency"),
                "current_joint_weight": row.get("joint_scenario_frequency"),
                "route_days": row.get("route_days"),
                "representative_date": row.get("representative_date"),
                "example_dates": (row.get("example_dates") or [])[:8],
                "distance_stats": distance_stats.get(str(vehicle_id), {}).get(int(row["cluster"]), {}),
                "role": "other_vehicle",
            }
        )

    for vehicle_id, cluster_result in image_clusters.items():
        known = {int(row["cluster"]) for row in options.get(str(vehicle_id), [])}
        for cluster in cluster_result.get("cluster_summary", []):
            if int(cluster["cluster"]) in known:
                continue
            options.setdefault(str(vehicle_id), []).append(
                {
                    "vehicle_id": int(vehicle_id),
                    "cluster": cluster["cluster"],
                    "current_route_weight": cluster.get("share"),
                    "route_days": cluster.get("days"),
                    "representative_date": cluster.get("representative_date"),
                    "example_dates": (cluster.get("example_dates") or [])[:8],
                    "distance_stats": distance_stats.get(str(vehicle_id), {}).get(int(cluster["cluster"]), {}),
                }
            )
    return options


def _compact_argument_context(scenario_summary: dict, scenario_summary_path: Path) -> dict:
    output_files = scenario_summary.get("output_files", {})
    argument_path_text = output_files.get("argument_correlations")
    if not argument_path_text:
        return {}
    argument_path = _resolve_path(argument_path_text, scenario_summary_path)
    if not argument_path.exists():
        return {}
    argument = _read_json(argument_path)
    target_vehicle = int(scenario_summary["inputs"]["target_vehicle"])
    rows = []
    for row in argument.get("pairwise_argument_correlations", []):
        if target_vehicle not in (int(row["vehicle_a"]), int(row["vehicle_b"])):
            continue
        rows.append(
            {
                "vehicle_a": row["vehicle_a"],
                "vehicle_b": row["vehicle_b"],
                "total_correlation": row.get("total_correlation"),
                "statistical_total_correlation": row.get("statistical_total_correlation"),
                "features": row.get("features"),
                "llm_rationale": row.get("llm_rationale"),
            }
        )
    return {
        "method": argument.get("method", {}),
        "target_pair_correlations": rows,
        "operational_evidence": argument.get("operational_evidence", {}),
    }


def _request_llm_route_weights(
    scenario_summary: dict,
    cluster_options: dict[str, list[dict]],
    argument_context: dict,
    provider: str,
    model_name: Optional[str],
) -> dict:
    llm, resolved_model = _build_llm(provider, model_name)
    payload = {
        "goal": (
            "Adjust route-cluster sampling weights for scenario generation. "
            "The current pipeline already selected truck appearance frequencies; "
            "you are improving route-cluster choice for each truck."
        ),
        "inputs": scenario_summary["inputs"],
        "truck_appearance_frequencies": scenario_summary.get("truck_appearance_frequencies", {}),
        "cluster_options_by_vehicle": cluster_options,
        "argument_context": argument_context,
    }
    prompt = f"""
You are improving route-cluster sampling for truck scenario generation.

Task:
- For each vehicle, assign route_cluster_weight values across its available clusters.
- Weights for each vehicle should be non-negative and sum to 1.0.
- Keep frequent clusters likely, but adjust weights when distance stats, zone behavior, or route evidence suggests another cluster better represents realistic scenario behavior.
- Do not choose clusters outside the provided cluster_options_by_vehicle.
- Return concise rationales.

Important:
- You are not changing which trucks are correlated. You are changing which route cluster is sampled once a truck appears.
- For other trucks, final joint probability will be truck_appearance_frequency * your route_cluster_weight.
- For the target truck, your route_cluster_weight directly controls target route sampling.

Input evidence:
{json.dumps(payload, indent=2)}

Return only this JSON schema:
{{
  "vehicle_route_cluster_weights": {{
    "155": [
      {{"cluster": 0, "route_cluster_weight": 0.0, "rationale": "short reason"}}
    ]
  }},
  "global_rationale": "short explanation",
  "risks": ["short risk"]
}}
"""
    parsed = _invoke_json(llm, prompt)
    parsed["llm_provider"] = provider
    parsed["llm_model"] = resolved_model
    return parsed


def _normalize_cluster_weights(
    llm_result: dict,
    cluster_options: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    output = {}
    raw_by_vehicle = llm_result.get("vehicle_route_cluster_weights", {})
    for vehicle_id, options in cluster_options.items():
        option_by_cluster = {int(option["cluster"]): option for option in options}
        raw_rows = raw_by_vehicle.get(vehicle_id, raw_by_vehicle.get(int(vehicle_id), []))
        weights = {}
        rationales = {}
        for row in raw_rows or []:
            try:
                cluster = int(row["cluster"])
                weight = max(0.0, float(row.get("route_cluster_weight", 0.0)))
            except (KeyError, TypeError, ValueError):
                continue
            if cluster in option_by_cluster:
                weights[cluster] = weight
                rationales[cluster] = row.get("rationale", "")

        if not weights or sum(weights.values()) == 0:
            weights = {
                int(option["cluster"]): max(0.0, float(option.get("current_route_weight") or 0.0))
                for option in options
            }
        if sum(weights.values()) == 0:
            weights = {int(option["cluster"]): 1.0 for option in options}

        total = sum(weights.values())
        output[vehicle_id] = [
            {
                **option,
                "llm_route_cluster_weight": weights.get(int(option["cluster"]), 0.0) / total,
                "llm_rationale": rationales.get(int(option["cluster"]), ""),
            }
            for option in options
        ]
    return output


def _sample_weighted(rows: list[dict], weight_key: str, rng: random.Random) -> dict:
    weights = [max(0.0, float(row.get(weight_key) or 0.0)) for row in rows]
    total = sum(weights)
    if total <= 0:
        return rng.choice(rows)
    threshold = rng.random() * total
    running = 0.0
    for row, weight in zip(rows, weights):
        running += weight
        if running >= threshold:
            return row
    return rows[-1]


def _sample_date(row: dict, rng: random.Random) -> Optional[str]:
    example_dates = row.get("example_dates") or []
    if example_dates:
        return rng.choice(example_dates)
    return row.get("representative_date")


def _sample_llm_route_cluster_scenario(
    scenario_summary: dict,
    adjusted_options: dict[str, list[dict]],
    scenario_days: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    target_vehicle = int(scenario_summary["inputs"]["target_vehicle"])
    other_vehicles = [int(vehicle_id) for vehicle_id in scenario_summary["inputs"]["other_vehicles"]]
    truck_frequencies = {
        int(vehicle_id): values["appearance_frequency"]
        for vehicle_id, values in scenario_summary.get("truck_appearance_frequencies", {}).items()
    }

    target_options = adjusted_options[str(target_vehicle)]
    other_options = []
    for vehicle_id in other_vehicles:
        for option in adjusted_options.get(str(vehicle_id), []):
            truck_frequency = truck_frequencies.get(vehicle_id, 0.0)
            other_options.append(
                {
                    **option,
                    "truck_appearance_frequency": truck_frequency,
                    "llm_joint_scenario_frequency": truck_frequency * option["llm_route_cluster_weight"],
                }
            )

    days = []
    for day_index in range(1, scenario_days + 1):
        target_route = _sample_weighted(target_options, "llm_route_cluster_weight", rng)
        other_route = _sample_weighted(other_options, "llm_joint_scenario_frequency", rng)
        days.append(
            {
                "scenario_day": day_index,
                "target_vehicle": {
                    "vehicle_id": target_vehicle,
                    "cluster": target_route["cluster"],
                    "sampled_route_date": _sample_date(target_route, rng),
                    "representative_date": target_route.get("representative_date"),
                    "route_cluster_weight": target_route["llm_route_cluster_weight"],
                    "sampling_weight": "llm_route_cluster_weight",
                    "llm_rationale": target_route.get("llm_rationale", ""),
                },
                "other_vehicle": {
                    "vehicle_id": other_route["vehicle_id"],
                    "cluster": other_route["cluster"],
                    "sampled_route_date": _sample_date(other_route, rng),
                    "representative_date": other_route.get("representative_date"),
                    "truck_appearance_frequency": other_route["truck_appearance_frequency"],
                    "route_cluster_weight": other_route["llm_route_cluster_weight"],
                    "joint_scenario_frequency": other_route["llm_joint_scenario_frequency"],
                    "sampling_weight": "llm_joint_scenario_frequency",
                    "llm_rationale": other_route.get("llm_rationale", ""),
                },
            }
        )
    return {
        "scenario_days_requested": scenario_days,
        "seed": seed,
        "sampling_method": (
            "LLM-adjusted route-cluster weights. Target routes sample from "
            "llm_route_cluster_weight; other routes sample from "
            "truck_appearance_frequency * llm_route_cluster_weight."
        ),
        "days": days,
    }


def build_llm_route_cluster_sampling(
    scenario_summary_path: Path,
    output_dir: Path,
    distance_file: Path = DEFAULT_DISTANCE_FILE,
    scenario_days: Optional[int] = None,
    seed: Optional[int] = None,
    llm_provider: str = "anthropic",
    llm_model: Optional[str] = None,
) -> dict:
    scenario_summary = _read_json(scenario_summary_path)
    output_files = scenario_summary.get("output_files", {})
    image_clusters_path = _resolve_path(output_files["image_clusters"], scenario_summary_path)
    image_clusters = _read_json(image_clusters_path)

    start_date = _normalize_date(scenario_summary["inputs"]["start_date"])
    end_date = _normalize_date(scenario_summary["inputs"]["end_date"])
    vehicles = [int(scenario_summary["inputs"]["target_vehicle"]), *map(int, scenario_summary["inputs"]["other_vehicles"])]
    distances = _load_distance_by_vehicle_date(distance_file, vehicles, start_date, end_date)
    distance_stats = _cluster_distance_stats(image_clusters, vehicles, distances)
    cluster_options = _build_cluster_options(scenario_summary, image_clusters, distance_stats)
    argument_context = _compact_argument_context(scenario_summary, scenario_summary_path)
    llm_result = _request_llm_route_weights(
        scenario_summary,
        cluster_options,
        argument_context,
        llm_provider,
        llm_model,
    )
    adjusted_options = _normalize_cluster_weights(llm_result, cluster_options)
    resolved_days = scenario_days or int(scenario_summary["inputs"]["scenario_days"])
    resolved_seed = seed if seed is not None else int(scenario_summary["inputs"]["seed"])
    final_scenario = _sample_llm_route_cluster_scenario(
        scenario_summary,
        adjusted_options,
        resolved_days,
        resolved_seed,
    )

    run_label = scenario_summary_path.parent.name + "_route_cluster_llm"
    run_dir = output_dir / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "inputs": {
            "scenario_summary": str(scenario_summary_path),
            "distance_file": str(distance_file),
            "scenario_days": resolved_days,
            "seed": resolved_seed,
            "llm_provider": llm_provider,
            "llm_model": llm_result.get("llm_model"),
        },
        "llm_route_weight_response": llm_result,
        "cluster_options_by_vehicle": cluster_options,
        "adjusted_cluster_options_by_vehicle": adjusted_options,
        "final_scenario": final_scenario,
        "output_files": {
            "summary": str(run_dir / "route_cluster_sampling_summary.json"),
            "route_cluster_weights": str(run_dir / "llm_route_cluster_weights.json"),
            "final_scenario": str(run_dir / "final_route_cluster_llm_scenario.json"),
        },
    }
    (run_dir / "llm_route_cluster_weights.json").write_text(
        json.dumps(adjusted_options, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "final_route_cluster_llm_scenario.json").write_text(
        json.dumps(final_scenario, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "route_cluster_sampling_summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use an LLM to adjust route-cluster sampling weights for an existing scenario run."
    )
    parser.add_argument("--scenario-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--distance-file", type=Path, default=DEFAULT_DISTANCE_FILE)
    parser.add_argument("--scenario-days", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--llm-provider", choices=("anthropic", "openai", "deepseek", "glm"), default="anthropic")
    parser.add_argument("--llm-model")
    args = parser.parse_args()

    result = build_llm_route_cluster_sampling(
        scenario_summary_path=args.scenario_summary,
        output_dir=args.output_dir,
        distance_file=args.distance_file,
        scenario_days=args.scenario_days,
        seed=args.seed,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )
    print(f"Wrote route-cluster LLM summary to {result['output_files']['summary']}")
    print(f"Wrote final scenario to {result['output_files']['final_scenario']}")
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
