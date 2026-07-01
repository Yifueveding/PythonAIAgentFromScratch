import json
from pathlib import Path

from main import _validate_scenario_distances


TARGET_VEHICLE = 155
OTHER_VEHICLES_20 = [
    1247,
    1311,
    1312,
    1350,
    1363,
    1421,
    1423,
    1506,
    1575,
    1576,
    1578,
    1667,
    1687,
    1688,
    1785,
    1993,
    2191,
    2201,
    2518,
]
VEHICLES_20 = [TARGET_VEHICLE, *OTHER_VEHICLES_20]
VALIDATION_DATES = [
    "2023-01-05",
    "2023-01-15",
    "2023-01-15",
    "2023-01-31",
    "2023-01-25",
]

GLOBAL_MODELS = [
    ("GPT 5.5", "openai", "gpt-5.5", "gpt_5_5"),
    ("Claude Haiku 4.5", "anthropic", "claude-haiku-4-5-20251001", "claude_haiku_4_5"),
    ("Claude Sonnet 5", "anthropic", "claude-sonnet-5", "claude_sonnet_5"),
    ("DeepSeek V4 Pro", "deepseek", "deepseek-v4-pro", "deepseek_v4_pro"),
]
NO_GLOBAL_MODELS = [
    ("GPT-4o Mini", "openai", "gpt-4o-mini", "gpt_4o_mini"),
    *GLOBAL_MODELS,
]
PREVIOUS_COMPARISON = Path("model_comparison_runs/comparison_5_9_20_vehicle_runs.json")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _row(
    label: str,
    provider: str,
    model: str,
    variant: str,
    result: dict,
    output_path: Path,
) -> dict:
    validation = _validate_scenario_distances(result, VALIDATION_DATES, VEHICLES_20)
    return {
        "model_label": label,
        "provider": provider,
        "model": model,
        "variant": variant,
        "status": "completed",
        "mean_individual_relative_distance_error": validation["mean_individual_relative_distance_error"],
        "fleet_total_relative_distance_error": validation["fleet_total_relative_distance_error"],
        "output_path": str(output_path),
        "validation": validation,
    }


def _scenario_row(
    root: Path,
    label: str,
    provider: str,
    model: str,
    slug: str,
    variant: str,
) -> dict:
    path = root / slug / "llm_reasoning_correlation" / "scenario_summary_llm.json"
    result = _read_json(path)
    return _row(label, provider, model, variant, result, path)


def _route_row(
    root: Path,
    label: str,
    provider: str,
    model: str,
    slug: str,
    variant: str,
) -> dict:
    path = (
        root
        / slug
        / "route_cluster_sampling_llm"
        / "llm_reasoning_correlation_route_cluster_llm"
        / "route_cluster_sampling_summary.json"
    )
    result = _read_json(path)
    return _row(label, provider, model, variant, {"final_scenario": result["final_scenario"]}, path)


def _previous_gpt4o_rows() -> list[dict]:
    previous = _read_json(PREVIOUS_COMPARISON)["runs"]["20_vehicles"]
    rows = []
    for source_key, variant in [
        ("llm_reasoning_correlation", "LLM_reasoning_correlation"),
        ("route_cluster_sampling_llm", "Route-cluster_sampling_LLM"),
    ]:
        path = Path(previous[source_key]["scenario_file"])
        result = {"final_scenario": _read_json(path)}
        rows.append(_row("GPT-4o Mini", "openai", "gpt-4o-mini", variant, result, path))
        rows[-1]["source"] = str(PREVIOUS_COMPARISON)
    return rows


def _baseline_row() -> dict:
    previous = _read_json(PREVIOUS_COMPARISON)["runs"]["20_vehicles"]
    path = Path(previous["non_llm"]["scenario_file"])
    validation = _validate_scenario_distances(
        {"final_scenario": _read_json(path)},
        VALIDATION_DATES,
        VEHICLES_20,
    )
    return {
        "model_label": "Statistical baseline",
        "provider": "statistical",
        "model": "non_llm",
        "variant": "Statistical baseline",
        "status": "completed",
        "mean_individual_relative_distance_error": validation["mean_individual_relative_distance_error"],
        "fleet_total_relative_distance_error": validation["fleet_total_relative_distance_error"],
        "output_path": str(path),
        "validation": validation,
    }


def _summary(root: Path, include_global_context: bool, rows: list[dict], model_specs: list[dict]) -> dict:
    return {
        "inputs": {
            "target_vehicle": TARGET_VEHICLE,
            "other_vehicles": OTHER_VEHICLES_20,
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "image_feature_method": "pca",
            "image_clusters": 5,
            "scenario_days": 5,
            "seed": 0,
            "validation_dates": VALIDATION_DATES,
            "previous_comparison_file": str(PREVIOUS_COMPARISON),
            "include_global_context": include_global_context,
            "validation_note": (
                "Corrected metric: non-sampled vehicles are excluded from distance "
                "error calculations instead of being assigned generated distance 0."
            ),
        },
        "model_specs": model_specs,
        "results": rows,
    }


def rebuild_global() -> dict:
    root = Path("llm_api_model_comparison_runs")
    rows = [_baseline_row(), *_previous_gpt4o_rows()]
    for label, provider, model, slug in GLOBAL_MODELS:
        rows.append(_scenario_row(root, label, provider, model, slug, "LLM_reasoning_correlation"))
        rows.append(_route_row(root, label, provider, model, slug, "Route-cluster_sampling_LLM"))
    model_specs = [
        {"label": "GPT-4o Mini", "provider": "openai", "model": "gpt-4o-mini"},
        *[
            {"label": label, "provider": provider, "model": model}
            for label, provider, model, _ in GLOBAL_MODELS
        ],
    ]
    summary = _summary(root, True, rows, model_specs)
    _write_json(root / "llm_api_20_vehicle_comparison.json", summary)
    return summary


def rebuild_no_global() -> dict:
    root = Path("llm_api_model_comparison_runs_no_global_context")
    rows = [_baseline_row()]
    for label, provider, model, slug in NO_GLOBAL_MODELS:
        rows.append(
            _scenario_row(
                root,
                label,
                provider,
                model,
                slug,
                "LLM_reasoning_correlation_no_global_context",
            )
        )
        rows.append(
            _route_row(
                root,
                label,
                provider,
                model,
                slug,
                "Route-cluster_sampling_LLM_no_global_context",
            )
        )
    model_specs = [
        {"label": label, "provider": provider, "model": model}
        for label, provider, model, _ in NO_GLOBAL_MODELS
    ]
    summary = _summary(root, False, rows, model_specs)
    _write_json(root / "llm_api_20_vehicle_comparison.json", summary)
    return summary


def _print_rows(name: str, summary: dict) -> None:
    print(name)
    for row in summary["results"]:
        print(
            f"{row['model']} | {row['variant']} | "
            f"mean={row['mean_individual_relative_distance_error']:.6f} | "
            f"fleet={row['fleet_total_relative_distance_error']:.6f}"
        )


def main() -> None:
    global_summary = rebuild_global()
    no_global_summary = rebuild_no_global()
    _print_rows("global", global_summary)
    _print_rows("no_global", no_global_summary)


if __name__ == "__main__":
    main()
