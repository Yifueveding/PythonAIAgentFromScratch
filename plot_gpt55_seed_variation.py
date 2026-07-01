import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from main import _sample_final_scenario_days, _validate_scenario_distances
from route_cluster_sampling_llm import _sample_llm_route_cluster_scenario


SEEDS = [1, 2, 3, 4, 5]
VALIDATION_DATES = [
    "2023-01-05",
    "2023-01-15",
    "2023-01-15",
    "2023-01-31",
    "2023-01-25",
]
CASES = {
    5: Path("llm_api_model_comparison_runs_gpt_5_5_global_seed_5_5_vehicles/gpt_5_5"),
    10: Path("llm_api_model_comparison_runs_gpt_5_5_global_seed_5_10_vehicles/gpt_5_5"),
    20: Path("llm_api_model_comparison_runs_gpt_5_5_global_seed_5/gpt_5_5"),
}
OUT_DIR = Path("gpt_5_5_global_seed_variation_plots")
METRICS = [
    ("mean_individual_relative_distance_error", "Mean individual relative distance error"),
    ("fleet_total_relative_distance_error", "Fleet total relative distance error"),
]
VARIANTS = ["LLM correlation", "LLM correlation + sampling"]


def _sample_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def _load_rows() -> list[dict]:
    rows = []
    for vehicle_count, root in CASES.items():
        scenario_summary_path = root / "llm_reasoning_correlation/scenario_summary_llm.json"
        route_summary_path = (
            root
            / "route_cluster_sampling_llm/llm_reasoning_correlation_route_cluster_llm/route_cluster_sampling_summary.json"
        )
        scenario_summary = json.loads(scenario_summary_path.read_text(encoding="utf-8"))
        route_summary = json.loads(route_summary_path.read_text(encoding="utf-8"))
        vehicles = [
            int(scenario_summary["inputs"]["target_vehicle"]),
            *map(int, scenario_summary["inputs"]["other_vehicles"]),
        ]
        scenario_days = int(scenario_summary["inputs"]["scenario_days"])

        for seed in SEEDS:
            llm_final = _sample_final_scenario_days(
                scenario_summary["target_route_appearance_frequencies"],
                scenario_summary["route_appearance_frequencies"],
                scenario_days,
                seed,
            )
            llm_validation = _validate_scenario_distances(
                {"final_scenario": llm_final},
                VALIDATION_DATES,
                vehicles,
            )
            rows.append(
                {
                    "vehicle_count": vehicle_count,
                    "variant": "LLM correlation",
                    "seed": seed,
                    "mean_individual_relative_distance_error": llm_validation[
                        "mean_individual_relative_distance_error"
                    ],
                    "fleet_total_relative_distance_error": llm_validation[
                        "fleet_total_relative_distance_error"
                    ],
                }
            )

            route_final = _sample_llm_route_cluster_scenario(
                scenario_summary,
                route_summary["adjusted_cluster_options_by_vehicle"],
                scenario_days,
                seed,
            )
            route_validation = _validate_scenario_distances(
                {"final_scenario": route_final},
                VALIDATION_DATES,
                vehicles,
            )
            rows.append(
                {
                    "vehicle_count": vehicle_count,
                    "variant": "LLM correlation + sampling",
                    "seed": seed,
                    "mean_individual_relative_distance_error": route_validation[
                        "mean_individual_relative_distance_error"
                    ],
                    "fleet_total_relative_distance_error": route_validation[
                        "fleet_total_relative_distance_error"
                    ],
                }
            )
    return rows


def _summarize(rows: list[dict]) -> dict:
    summary = {}
    for vehicle_count in CASES:
        summary[str(vehicle_count)] = {}
        for variant in VARIANTS:
            subset = [
                row
                for row in rows
                if row["vehicle_count"] == vehicle_count and row["variant"] == variant
            ]
            summary[str(vehicle_count)][variant] = {}
            for metric, _ in METRICS:
                values = [row[metric] for row in subset]
                mean = sum(values) / len(values)
                summary[str(vehicle_count)][variant][metric] = {
                    "mean": mean,
                    "sample_std": _sample_std(values),
                    "min": min(values),
                    "max": max(values),
                    "values": values,
                }
    return summary


def _write_outputs(rows: list[dict], summary: dict) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "gpt_5_5_global_5_seed_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUT_DIR / "gpt_5_5_global_5_seed_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "seeds": SEEDS,
                "validation_dates": VALIDATION_DATES,
                "summary": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def _plot(rows: list[dict]) -> Path:
    colors = {"LLM correlation": "#4C78A8", "LLM correlation + sampling": "#F58518"}
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 18,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    case_labels = list(CASES.keys())

    for ax, (metric, ylabel) in zip(axes, METRICS):
        positions = []
        data = []
        box_colors = []
        tick_positions = []
        tick_labels = []
        for index, vehicle_count in enumerate(case_labels, start=1):
            base = index * 3
            tick_positions.append(base + 0.35)
            tick_labels.append(str(vehicle_count))
            for offset, variant in [(0.0, "LLM correlation"), (0.7, "LLM correlation + sampling")]:
                positions.append(base + offset)
                data.append(
                    [
                        row[metric]
                        for row in rows
                        if row["vehicle_count"] == vehicle_count and row["variant"] == variant
                    ]
                )
                box_colors.append(colors[variant])

        box = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True, showfliers=True)
        for patch, color in zip(box["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for median in box["medians"]:
            median.set_color("black")
        for pos, values, color in zip(positions, data, box_colors):
            mean = sum(values) / len(values)
            ax.errorbar(
                pos,
                mean,
                yerr=[[mean - min(values)], [max(values) - mean]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=4,
                markersize=5,
            )

        ax.set_title(ylabel)
        ax.set_xlabel("Number of vehicles")
        ax.set_ylabel("Error")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[variant],
            marker="s",
            linestyle="",
            markersize=10,
            alpha=0.6,
            label=variant,
        )
        for variant in VARIANTS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plot_path = OUT_DIR / "gpt_5_5_global_5_seed_boxplot.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> None:
    rows = _load_rows()
    summary = _summarize(rows)
    csv_path, json_path = _write_outputs(rows, summary)
    plot_path = _plot(rows)
    print(plot_path)
    print(csv_path)
    print(json_path)
    for vehicle_count in CASES:
        for variant in VARIANTS:
            mean_metric = summary[str(vehicle_count)][variant][
                "mean_individual_relative_distance_error"
            ]
            fleet_metric = summary[str(vehicle_count)][variant][
                "fleet_total_relative_distance_error"
            ]
            print(
                f"{vehicle_count} | {variant} | "
                f"mean_individual={mean_metric['mean']:.6f} +/- {mean_metric['sample_std']:.6f} | "
                f"fleet={fleet_metric['mean']:.6f} +/- {fleet_metric['sample_std']:.6f}"
            )


if __name__ == "__main__":
    main()
