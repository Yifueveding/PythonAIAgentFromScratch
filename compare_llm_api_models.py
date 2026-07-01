import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from main import _validate_scenario_distances
from main_llm import build_llm_scenario_generation
from route_cluster_sampling_llm import build_llm_route_cluster_sampling


DEFAULT_TARGET_VEHICLE = 155
DEFAULT_OTHER_VEHICLES = [
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
DEFAULT_VALIDATION_DATES = [
    "2023-01-05",
    "2023-01-15",
    "2023-01-15",
    "2023-01-31",
    "2023-01-25",
]
DEFAULT_MODEL_SPECS = [
    {
        "label": "GPT 5.5",
        "provider": "openai",
        "model": "gpt-5.5",
        "required_env": "OPENAI_API_KEY",
    },
    {
        "label": "Claude Haiku 4.5",
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "required_env": "ANTHROPIC_API_KEY",
    },
    {
        "label": "Claude Sonnet 5",
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "required_env": "ANTHROPIC_API_KEY",
    },
    {
        "label": "GLM 4.5 Air",
        "provider": "glm",
        "model": "glm-4.5-air",
        "required_env": "GLM_API_KEY|ZAI_API_KEY|ZHIPUAI_API_KEY",
    },
    {
        "label": "DeepSeek V4 Pro",
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "required_env": "DEEPSEEK_API_KEY",
    },
]
DEFAULT_PREVIOUS_COMPARISON_FILE = Path("model_comparison_runs/comparison_5_9_20_vehicle_runs.json")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _env_available(required_env: str) -> bool:
    load_dotenv(".env", override=False)
    load_dotenv("sample.env", override=False)
    load_dotenv("sample1.env", override=True)
    load_dotenv("sample2.env", override=True)
    load_dotenv("sample3.env", override=True)
    return any(os.getenv(key.strip()) for key in required_env.split("|"))


def _validation_row(model_spec: dict, model_variant: str, result: dict, validation: dict, output_path: str) -> dict:
    return {
        "model_label": model_spec["label"],
        "provider": model_spec["provider"],
        "model": result["inputs"].get("llm_model", model_spec["model"]),
        "variant": model_variant,
        "status": "completed",
        "mean_individual_relative_distance_error": validation["mean_individual_relative_distance_error"],
        "fleet_total_relative_distance_error": validation["fleet_total_relative_distance_error"],
        "output_path": output_path,
        "validation": validation,
    }


def _openai_before_rows(previous_comparison_file: Path) -> list[dict]:
    if not previous_comparison_file.exists():
        return [
            {
                "model_label": "OPEN_AI_before",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "variant": "LLM_reasoning_correlation",
                "status": "skipped",
                "reason": f"Missing previous comparison file: {previous_comparison_file}",
            },
            {
                "model_label": "OPEN_AI_before",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "variant": "Route-cluster_sampling_LLM",
                "status": "skipped",
                "reason": f"Missing previous comparison file: {previous_comparison_file}",
            },
        ]

    comparison = json.loads(previous_comparison_file.read_text(encoding="utf-8"))
    runs = comparison.get("runs", {}).get("20_vehicles", {})
    variant_map = {
        "llm_reasoning_correlation": "LLM_reasoning_correlation",
        "route_cluster_sampling_llm": "Route-cluster_sampling_LLM",
    }
    rows = []
    for source_key, variant in variant_map.items():
        source = runs.get(source_key)
        if not source:
            rows.append(
                {
                    "model_label": "OPEN_AI_before",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "variant": variant,
                    "status": "skipped",
                    "reason": f"Missing 20_vehicles/{source_key} in {previous_comparison_file}",
                }
            )
            continue
        rows.append(
            {
                "model_label": "OPEN_AI_before",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "variant": variant,
                "status": "completed",
                "mean_individual_relative_distance_error": source["mean_individual_relative_distance_error"],
                "fleet_total_relative_distance_error": source["fleet_total_relative_distance_error"],
                "output_path": source["scenario_file"],
                "source": str(previous_comparison_file),
                "note": "Reused from the earlier committed 20-vehicle OpenAI run.",
            }
        )
    return rows


def run_model_experiment(
    model_spec: dict,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict]:
    llm_variant = "LLM_reasoning_correlation"
    route_variant = "Route-cluster_sampling_LLM"
    if not args.include_global_context:
        llm_variant = "LLM_reasoning_correlation_no_global_context"
        route_variant = "Route-cluster_sampling_LLM_no_global_context"

    if not _env_available(model_spec["required_env"]):
        return [
            {
                "model_label": model_spec["label"],
                "provider": model_spec["provider"],
                "model": model_spec["model"],
                "variant": llm_variant,
                "status": "skipped",
                "reason": f"Missing required environment variable: {model_spec['required_env']}",
            },
            {
                "model_label": model_spec["label"],
                "provider": model_spec["provider"],
                "model": model_spec["model"],
                "variant": route_variant,
                "status": "skipped",
                "reason": f"Missing required environment variable: {model_spec['required_env']}",
            },
        ]

    model_slug = _slug(model_spec["label"])
    run_dir = output_dir / model_slug / "llm_reasoning_correlation"
    route_output_dir = output_dir / model_slug / "route_cluster_sampling_llm"
    vehicles = [args.target_vehicle, *args.other_vehicles]

    try:
        llm_result = build_llm_scenario_generation(
            target_vehicle=args.target_vehicle,
            other_vehicles=args.other_vehicles,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=output_dir,
            image_feature_method=args.image_feature_method,
            image_clusters=args.image_clusters,
            scenario_days=args.scenario_days,
            seed=args.seed,
            run_dir=run_dir,
            llm_provider=model_spec["provider"],
            llm_model=model_spec["model"],
            include_global_context=args.include_global_context,
        )
        llm_validation = _validate_scenario_distances(
            llm_result,
            args.validation_dates,
            vehicles,
        )
        rows = [
            _validation_row(
                model_spec,
                llm_variant,
                llm_result,
                llm_validation,
                llm_result["output_files"]["summary"],
            )
        ]

        route_result = build_llm_route_cluster_sampling(
            scenario_summary_path=Path(llm_result["output_files"]["summary"]),
            output_dir=route_output_dir,
            scenario_days=args.scenario_days,
            seed=args.seed,
            llm_provider=model_spec["provider"],
            llm_model=model_spec["model"],
            include_global_context=args.include_global_context,
        )
        route_validation = _validate_scenario_distances(
            {"final_scenario": route_result["final_scenario"]},
            args.validation_dates,
            vehicles,
        )
        rows.append(
            _validation_row(
                model_spec,
                route_variant,
                route_result,
                route_validation,
                route_result["output_files"]["summary"],
            )
        )
        return rows
    except Exception as exc:
        return [
            {
                "model_label": model_spec["label"],
                "provider": model_spec["provider"],
                "model": model_spec["model"],
                "variant": llm_variant,
                "status": "failed",
                "reason": str(exc),
            },
            {
                "model_label": model_spec["label"],
                "provider": model_spec["provider"],
                "model": model_spec["model"],
                "variant": route_variant,
                "status": "not_run",
                "reason": "Route-cluster sampling depends on a completed LLM reasoning correlation run.",
            },
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 20-vehicle LLM correlation and route-cluster sampling across LLM APIs."
    )
    parser.add_argument("--target-vehicle", type=int, default=DEFAULT_TARGET_VEHICLE)
    parser.add_argument("--other-vehicles", nargs="+", type=int, default=DEFAULT_OTHER_VEHICLES)
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2023-01-31")
    parser.add_argument("--image-feature-method", choices=("pca", "autoencoder"), default="pca")
    parser.add_argument("--image-clusters", type=int, default=5)
    parser.add_argument("--scenario-days", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-dates", nargs="+", default=DEFAULT_VALIDATION_DATES)
    parser.add_argument("--output-dir", type=Path, default=Path("llm_api_model_comparison_runs"))
    parser.add_argument("--previous-comparison-file", type=Path, default=DEFAULT_PREVIOUS_COMPARISON_FILE)
    parser.add_argument("--exclude-openai-before", action="store_true")
    parser.add_argument(
        "--no-global-context",
        dest="include_global_context",
        action="store_false",
        help=(
            "Run an ablation where LLM prompts do not receive global contextual "
            "metadata, representative routes, operational evidence, or compact "
            "argument context."
        ),
    )
    parser.set_defaults(include_global_context=True)
    parser.add_argument(
        "--model-specs",
        type=Path,
        help="Optional JSON file with rows containing label, provider, model, and required_env.",
    )
    parser.add_argument("--only-label", action="append", default=[])
    args = parser.parse_args()

    model_specs = DEFAULT_MODEL_SPECS
    if args.model_specs:
        model_specs = json.loads(args.model_specs.read_text(encoding="utf-8"))
    if args.only_label:
        wanted = set(args.only_label)
        model_specs = [model_spec for model_spec in model_specs if model_spec["label"] in wanted]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if not args.exclude_openai_before:
        rows.extend(_openai_before_rows(args.previous_comparison_file))
    for model_spec in model_specs:
        rows.extend(run_model_experiment(model_spec, args, args.output_dir))

    summary = {
        "inputs": {
            "target_vehicle": args.target_vehicle,
            "other_vehicles": args.other_vehicles,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "image_feature_method": args.image_feature_method,
            "image_clusters": args.image_clusters,
            "scenario_days": args.scenario_days,
            "seed": args.seed,
            "validation_dates": args.validation_dates,
            "previous_comparison_file": str(args.previous_comparison_file),
            "include_global_context": args.include_global_context,
        },
        "model_specs": model_specs,
        "results": rows,
    }
    output_file = args.output_dir / "llm_api_20_vehicle_comparison.json"
    output_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote API comparison to {output_file}")
    for row in rows:
        if row["status"] == "completed":
            print(
                f"{row['model_label']} / {row['variant']}: "
                f"mean={row['mean_individual_relative_distance_error']:.6f}, "
                f"fleet={row['fleet_total_relative_distance_error']:.6f}"
            )
        else:
            print(f"{row['model_label']} / {row['variant']}: {row['status']} - {row['reason']}")


if __name__ == "__main__":
    main()
