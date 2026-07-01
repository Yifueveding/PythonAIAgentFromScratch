import json
from pathlib import Path


TABLES = [
    (
        "global",
        "llm_api_model_comparison_runs/llm_api_20_vehicle_comparison.json",
        "Benchmark against non-LLM scenario generation for the 20-vehicle case with global contextual information",
        "tab:llm-api-benchmark-global",
    ),
    (
        "no_global",
        "llm_api_model_comparison_runs_no_global_context/llm_api_20_vehicle_comparison.json",
        "Benchmark against non-LLM scenario generation for the 20-vehicle case without global contextual information",
        "tab:llm-api-benchmark-no-global",
    ),
    (
        "no_global_seed_5",
        "llm_api_model_comparison_runs_no_global_context_seed_5/llm_api_20_vehicle_comparison.json",
        "Benchmark against non-LLM scenario generation for the 20-vehicle case without global contextual information using random seed 5",
        "tab:llm-api-benchmark-no-global-seed-5",
    ),
]
BENCHMARK_SUMMARY = Path("gpt_5_5_global_seed_5_non_llm_benchmarks/non_llm_benchmark_summary.json")
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


def _benchmark_20_vehicle_baseline() -> tuple[float, float]:
    summary = json.loads(BENCHMARK_SUMMARY.read_text(encoding="utf-8"))
    row = next(row for row in summary["runs"] if row["vehicle_count"] == 20)
    return (
        row["mean_individual_relative_distance_error"],
        row["fleet_total_relative_distance_error"],
    )


def _percent(value: float, baseline: float) -> float:
    return (value - baseline) / baseline * 100


def main() -> None:
    baseline_mean, baseline_fleet = _benchmark_20_vehicle_baseline()
    for title, path, caption, label in TABLES:
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        print(f"% {title}")
        print(f"% baseline source={BENCHMARK_SUMMARY}")
        print(f"% baseline mean={baseline_mean:.6f}, fleet={baseline_fleet:.6f}")
        print(r"\begin{table}[ht]")
        print(r"\centering")
        print(f"\\caption{{{caption}}}")
        print(f"\\label{{{label}}}")
        print(r"\begin{tabular}{l l c c}")
        print(r"\hline")
        print(r"\textbf{Model ID} & \textbf{Method} & \textbf{Mean Individual Error} & \textbf{Fleet Total Error} \\")
        print(r"\hline")
        print(f"-- & Statistical baseline & {baseline_mean:.2f} & {baseline_fleet:.2f} \\\\")
        print(r"\hline")
        for row in summary["results"]:
            if row["variant"] == "Statistical baseline":
                continue
            if row["status"] != "completed":
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
            if method == "LLM correlation+sample":
                print(r"\hline")
        print(r"\end{tabular}")
        print(r"\end{table}")
        print()


if __name__ == "__main__":
    main()
