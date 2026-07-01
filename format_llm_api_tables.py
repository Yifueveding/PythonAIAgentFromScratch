import json


TABLES = [
    ("global", "llm_api_model_comparison_runs/llm_api_20_vehicle_comparison.json"),
    ("no_global", "llm_api_model_comparison_runs_no_global_context/llm_api_20_vehicle_comparison.json"),
]
MODEL_LABELS = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-5.5": "gpt-5.5",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5",
    "claude-sonnet-5": "claude-sonnet-5",
    "deepseek-v4-pro": "deepseek-v4-pro",
}
METHOD_LABELS = {
    "LLM_reasoning_correlation": "LLM correlation",
    "Route-cluster_sampling_LLM": "LLM correlation+sample",
    "LLM_reasoning_correlation_no_global_context": "LLM correlation",
    "Route-cluster_sampling_LLM_no_global_context": "LLM correlation+sample",
}


def _percent(value: float, baseline: float) -> float:
    return (value - baseline) / baseline * 100


def main() -> None:
    for title, path in TABLES:
        summary = json.load(open(path, encoding="utf-8"))
        baseline = next(row for row in summary["results"] if row["variant"] == "Statistical baseline")
        baseline_mean = baseline["mean_individual_relative_distance_error"]
        baseline_fleet = baseline["fleet_total_relative_distance_error"]
        print(f"% {title}")
        print(f"% baseline mean={baseline_mean:.6f}, fleet={baseline_fleet:.6f}")
        for row in summary["results"]:
            if row["variant"] == "Statistical baseline":
                continue
            model = MODEL_LABELS[row["model"]]
            method = METHOD_LABELS[row["variant"]]
            mean = row["mean_individual_relative_distance_error"]
            fleet = row["fleet_total_relative_distance_error"]
            print(
                f"\\texttt{{{model}}} & {method} "
                f"& {mean:.2f} ({_percent(mean, baseline_mean):+.2f}\\%) "
                f"& {fleet:.2f} ({_percent(fleet, baseline_fleet):+.2f}\\%) \\\\"
            )
        print()


if __name__ == "__main__":
    main()
