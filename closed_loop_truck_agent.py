import argparse
import csv
import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from multi_truck_scenario import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_PASSES_FILE,
    _image_similarity,
    _load_image_features,
    _load_vehicle_passes,
    _pass_correlation,
)


DEFAULT_OUTPUT_ROOT = Path("closed_loop_runs")

DEFAULT_WEIGHTS = {
    "pass_score": 0.4,
    "image_similarity": 0.4,
    "purpose_match": 0.1,
    "temperature_similarity": 0.1,
}

WEIGHT_KEYS = tuple(DEFAULT_WEIGHTS)


def _date_in_window(date: str, start_date: str, end_date: str) -> bool:
    return start_date <= date <= end_date


def _filter_by_date(data: dict[str, object], start_date: str, end_date: str) -> dict[str, object]:
    return {
        date: value
        for date, value in data.items()
        if _date_in_window(date, start_date, end_date)
    }


def _load_temperature(path: Optional[Path], target_year: Optional[int] = None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(line for line in file if not line.startswith("#"))
        if reader.fieldnames is None:
            return {}

        date_column = None
        for candidate in ("date", "local_time", "time"):
            if candidate in reader.fieldnames:
                date_column = candidate
                break

        if date_column is None:
            raise ValueError(f"{path} needs a date, local_time, or time column.")

        temp_column = None
        for candidate in ("temperature", "temp", "temperature_f", "temperature_c", "t2m"):
            if candidate in reader.fieldnames:
                temp_column = candidate
                break

        if temp_column is None:
            raise ValueError(f"{path} needs a temperature column.")

        daily_temperatures: dict[str, list[float]] = {}
        for row in reader:
            raw_date = row.get(date_column)
            raw_temperature = row.get(temp_column)
            if not raw_date or raw_temperature in ("", None):
                continue

            date = raw_date[:10]
            if target_year is not None:
                date = f"{target_year}{date[4:]}"
            daily_temperatures.setdefault(date, []).append(float(raw_temperature))

        return {
            date: sum(values) / len(values)
            for date, values in daily_temperatures.items()
        }


def _load_metadata(path: Optional[Path]) -> dict[int, dict[str, str]]:
    if path is None or not path.exists():
        return {}

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return {}

        vehicle_column = "VehicleId" if "VehicleId" in reader.fieldnames else "vehicle_id"
        return {
            int(row[vehicle_column]): row
            for row in reader
            if row.get(vehicle_column)
        }


def _load_feedback(path: Optional[Path]) -> dict[tuple[int, int], float]:
    if path is None or not path.exists():
        return {}

    feedback = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return {}

        for row in reader:
            left = int(row.get("vehicle_a") or row.get("VehicleA"))
            right = int(row.get("vehicle_b") or row.get("VehicleB"))
            if "feedback_score" in row and row["feedback_score"] != "":
                label = float(row["feedback_score"])
            elif "label" in row:
                label = 1.0 if row["label"].lower() in ("1", "true", "related", "shared_route") else 0.0
            else:
                continue
            feedback[tuple(sorted((left, right)))] = max(0.0, min(1.0, label))
    return feedback


def _average_temperature_similarity(
    left_dates: set[str],
    right_dates: set[str],
    temperatures: dict[str, float],
) -> tuple[Optional[float], Optional[float], int]:
    if not temperatures:
        return None, None, 0

    left_values = [temperatures[date] for date in left_dates if date in temperatures]
    right_values = [temperatures[date] for date in right_dates if date in temperatures]
    if not left_values or not right_values:
        return None, None, 0

    left_avg = sum(left_values) / len(left_values)
    right_avg = sum(right_values) / len(right_values)
    similarity = 1.0 - min(abs(left_avg - right_avg) / 50.0, 1.0)
    return similarity, (left_avg + right_avg) / 2.0, min(len(left_values), len(right_values))


def _purpose_match(left_id: int, right_id: int, metadata: dict[int, dict[str, str]]) -> Optional[float]:
    if not metadata:
        return None

    left = metadata.get(left_id, {})
    right = metadata.get(right_id, {})
    purpose_left = left.get("purpose") or left.get("truck_purpose")
    purpose_right = right.get("purpose") or right.get("truck_purpose")
    if not purpose_left or not purpose_right:
        return None
    return 1.0 if purpose_left == purpose_right else 0.0


def _score_features(features: dict, weights: dict[str, float]) -> Optional[float]:
    weighted_values = []
    for key, weight in weights.items():
        value = features.get(key)
        if value is None:
            continue
        weighted_values.append((float(value), float(weight)))

    if not weighted_values:
        return None

    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight == 0:
        return None
    return sum(value * weight for value, weight in weighted_values) / total_weight


def _scenario_label(score: Optional[float]) -> str:
    if score is None:
        return "insufficient_data"
    if score >= 0.75:
        return "shared_route_behavior"
    if score >= 0.55:
        return "partially_shared_route_behavior"
    return "mostly_independent_route_behavior"


def _stability_label(delta: Optional[float], tolerance: float) -> str:
    if delta is None:
        return "insufficient_data"
    if abs(delta) <= tolerance:
        return "stable_close_to_training"
    if delta > 0:
        return "stronger_than_training"
    return "weaker_than_training"


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    normalized_input = {
        key: max(0.0, float(weights.get(key, 0.0)))
        for key in WEIGHT_KEYS
    }
    total = sum(normalized_input.values())
    if total == 0:
        return dict(DEFAULT_WEIGHTS)
    return {key: value / total for key, value in normalized_input.items()}


def _learn_weights(rows: list[dict], feedback: dict[tuple[int, int], float]) -> dict[str, float]:
    labeled_rows = [
        row
        for row in rows
        if tuple(sorted((row["vehicle_a"], row["vehicle_b"]))) in feedback
    ]
    if not labeled_rows:
        return dict(DEFAULT_WEIGHTS)

    feature_names = list(DEFAULT_WEIGHTS)
    weights = {name: 0.0 for name in feature_names}
    positive_count = 0.0

    for row in labeled_rows:
        label = feedback[tuple(sorted((row["vehicle_a"], row["vehicle_b"])))]
        positive_count += label
        for name in feature_names:
            value = row["features"].get(name)
            if value is not None:
                weights[name] += label * float(value)

    if positive_count == 0:
        return dict(DEFAULT_WEIGHTS)

    weights = {name: value / positive_count for name, value in weights.items()}
    return _normalize_weights(weights)


def _build_pair_features(
    vehicle_ids: list[int],
    image_features: dict[int, dict[str, np.ndarray]],
    passes_by_vehicle: dict[int, dict[str, tuple[int, ...]]],
    temperatures: dict[str, float],
    metadata: dict[int, dict[str, str]],
    start_date: str,
    end_date: str,
    weights: dict[str, float],
) -> list[dict]:
    rows = []

    for left_id, right_id in combinations(vehicle_ids, 2):
        left_images = _filter_by_date(image_features[left_id], start_date, end_date)
        right_images = _filter_by_date(image_features[right_id], start_date, end_date)
        left_passes = _filter_by_date(passes_by_vehicle.get(left_id, {}), start_date, end_date)
        right_passes = _filter_by_date(passes_by_vehicle.get(right_id, {}), start_date, end_date)

        pass_corr, common_pass_dates, pass_features = _pass_correlation(left_passes, right_passes)
        pass_score = None if pass_corr is None else max(0.0, min(1.0, (pass_corr + 1.0) / 2.0))
        image_sim, common_image_dates, similar_dates = _image_similarity(left_images, right_images)
        temp_sim, average_temperature, temp_dates = _average_temperature_similarity(
            set(left_passes) | set(left_images),
            set(right_passes) | set(right_images),
            temperatures,
        )
        purpose = _purpose_match(left_id, right_id, metadata)

        features = {
            "pass_score": pass_score,
            "pass_correlation": pass_corr,
            "image_similarity": image_sim,
            "purpose_match": purpose,
            "temperature_similarity": temp_sim,
        }
        score = _score_features(features, weights)

        rows.append(
            {
                "vehicle_a": left_id,
                "vehicle_b": right_id,
                "features": features,
                "learned_similarity": score,
                "scenario_label": _scenario_label(score),
                "common_pass_dates": common_pass_dates,
                "pass_features_compared": pass_features,
                "common_image_dates": common_image_dates,
                "temperature_dates": temp_dates,
                "average_temperature": average_temperature,
                "most_similar_image_dates": similar_dates,
            }
        )

    rows.sort(
        key=lambda row: row["learned_similarity"] if row["learned_similarity"] is not None else -1.0,
        reverse=True,
    )
    return rows


def _pair_key(row: dict) -> tuple[int, int]:
    return tuple(sorted((row["vehicle_a"], row["vehicle_b"])))


def _build_performance_vs_train(split_results: dict, tolerance: float = 0.05) -> dict:
    train_rows = {
        _pair_key(row): row
        for row in split_results["train"]["pairwise_relationships"]
    }
    performance = {}

    for split_name in ("validation", "test"):
        rows = []
        for row in split_results[split_name]["pairwise_relationships"]:
            train_row = train_rows.get(_pair_key(row))
            train_score = None if train_row is None else train_row["learned_similarity"]
            split_score = row["learned_similarity"]
            delta = None
            absolute_delta = None
            if train_score is not None and split_score is not None:
                delta = split_score - train_score
                absolute_delta = abs(delta)

            rows.append(
                {
                    "vehicle_a": row["vehicle_a"],
                    "vehicle_b": row["vehicle_b"],
                    "train_similarity": train_score,
                    f"{split_name}_similarity": split_score,
                    f"{split_name}_delta_from_train": delta,
                    f"{split_name}_absolute_delta_from_train": absolute_delta,
                    "stability_label": _stability_label(delta, tolerance),
                    "target": "smaller absolute delta means better closed-loop stability",
                }
            )

        rows.sort(
            key=lambda item: (
                item[f"{split_name}_absolute_delta_from_train"]
                if item[f"{split_name}_absolute_delta_from_train"] is not None
                else float("inf")
            )
        )
        valid_deltas = [
            item[f"{split_name}_absolute_delta_from_train"]
            for item in rows
            if item[f"{split_name}_absolute_delta_from_train"] is not None
        ]
        performance[split_name] = {
            "goal": "Keep calculated split correlation close to training correlation.",
            "tolerance": tolerance,
            "mean_absolute_delta_from_train": (
                sum(valid_deltas) / len(valid_deltas) if valid_deltas else None
            ),
            "pairwise": rows,
        }

    return performance


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


def _summarize_for_llm(run: dict) -> dict:
    return {
        "vehicles": run["vehicles"],
        "current_weights": run["learned_weights"],
        "performance_goal": run["performance_goal"],
        "feature_meaning": run["feature_meaning"],
        "performance_vs_train": run["performance_vs_train"],
        "split_pair_features": {
            split_name: [
                {
                    "vehicle_a": row["vehicle_a"],
                    "vehicle_b": row["vehicle_b"],
                    "learned_similarity": row["learned_similarity"],
                    "features": row["features"],
                    "common_pass_dates": row["common_pass_dates"],
                    "common_image_dates": row["common_image_dates"],
                    "scenario_label": row["scenario_label"],
                }
                for row in split["pairwise_relationships"]
            ]
            for split_name, split in run["splits"].items()
        },
    }


def _request_llm_weight_update(
    run: dict,
    model_name: Optional[str] = None,
    provider: str = "anthropic",
) -> dict:
    load_dotenv()
    load_dotenv("sample.env", override=False)
    load_dotenv("sample1.env", override=True)
    for env_key in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        if os.getenv(env_key):
            os.environ[env_key] = os.environ[env_key].strip()

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise RuntimeError("langchain-anthropic is required for --use-llm.") from e

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for --use-llm.")
        resolved_model = model_name or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        llm = ChatAnthropic(model=resolved_model)
    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise RuntimeError("langchain-openai is required for --llm-provider openai.") from e

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for --llm-provider openai.")
        resolved_model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=resolved_model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    payload = _summarize_for_llm(run)
    prompt = f"""
You are updating feature weights for a closed-loop truck correlation system.

Goal:
- The test-period truck correlation should stay as close as possible to the training-period baseline.
- Smaller absolute delta from training means better closed-loop performance.
- Validation can be used as a tuning signal, but test stability is the final goal.

Allowed weight keys:
{list(WEIGHT_KEYS)}

Rules:
- Return only one JSON object.
- Weight values must be non-negative and should sum to 1.0.
- If a feature is missing/null for most rows, reduce its weight.
- Preserve explainability; do not invent unavailable features.
- Include concise rationale and risks.

Input evidence:
{json.dumps(payload, indent=2)}

Return this schema:
{{
  "updated_weights": {{
    "pass_score": 0.0,
    "image_similarity": 0.0,
    "purpose_match": 0.0,
    "temperature_similarity": 0.0
  }},
  "rationale": "short explanation",
  "expected_effect": "short explanation",
  "risks": ["short risk"]
}}
"""
    response = llm.invoke(prompt)
    parsed = _extract_json_object(_llm_json_text(response))
    updated_weights = _normalize_weights(parsed.get("updated_weights", {}))
    return {
        "provider": provider,
        "model": resolved_model,
        "updated_weights": updated_weights,
        "rationale": parsed.get("rationale", ""),
        "expected_effect": parsed.get("expected_effect", ""),
        "risks": parsed.get("risks", []),
    }


def _rescore_split_results(split_results: dict, weights: dict[str, float]) -> dict:
    rescored = json.loads(json.dumps(split_results))
    for split in rescored.values():
        for row in split["pairwise_relationships"]:
            row["learned_similarity"] = _score_features(row["features"], weights)
            row["scenario_label"] = _scenario_label(row["learned_similarity"])
        split["pairwise_relationships"].sort(
            key=lambda row: row["learned_similarity"] if row["learned_similarity"] is not None else -1.0,
            reverse=True,
        )
    return rescored


def _apply_llm_weight_update(
    run: dict,
    model_name: Optional[str] = None,
    provider: str = "anthropic",
) -> dict:
    update = _request_llm_weight_update(run, model_name, provider)
    llm_split_results = _rescore_split_results(run["splits"], update["updated_weights"])
    run["llm_weight_update"] = update
    run["llm_rescored_splits"] = llm_split_results
    run["llm_performance_vs_train"] = _build_performance_vs_train(llm_split_results)
    return run


def build_closed_loop_run(
    vehicle_ids: list[int],
    image_root: Path,
    passes_file: Path,
    temperature_file: Optional[Path],
    temperature_target_year: Optional[int],
    metadata_file: Optional[Path],
    feedback_file: Optional[Path],
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
    output_root: Path,
    use_llm: bool = False,
    llm_model: Optional[str] = None,
    llm_provider: str = "anthropic",
) -> dict:
    passes_by_vehicle, _ = _load_vehicle_passes(passes_file)
    image_features = {
        vehicle_id: _load_image_features(image_root, vehicle_id)
        for vehicle_id in vehicle_ids
    }
    temperatures = _load_temperature(temperature_file, temperature_target_year)
    metadata = _load_metadata(metadata_file)
    feedback = _load_feedback(feedback_file)

    train_rows_for_learning = _build_pair_features(
        vehicle_ids,
        image_features,
        passes_by_vehicle,
        temperatures,
        metadata,
        train_start,
        train_end,
        DEFAULT_WEIGHTS,
    )
    learned_weights = _learn_weights(train_rows_for_learning, feedback)

    splits = {
        "train": (train_start, train_end),
        "validation": (validation_start, validation_end),
        "test": (test_start, test_end),
    }
    split_results = {}
    for split_name, (start_date, end_date) in splits.items():
        split_results[split_name] = {
            "start_date": start_date,
            "end_date": end_date,
            "pairwise_relationships": _build_pair_features(
                vehicle_ids,
                image_features,
                passes_by_vehicle,
                temperatures,
                metadata,
                start_date,
                end_date,
                learned_weights,
            ),
        }

    run = {
        "vehicles": vehicle_ids,
        "inputs": {
            "image_root": str(image_root),
            "passes_file": str(passes_file),
            "temperature_file": None if temperature_file is None else str(temperature_file),
            "temperature_target_year": temperature_target_year,
            "metadata_file": None if metadata_file is None else str(metadata_file),
            "feedback_file": None if feedback_file is None else str(feedback_file),
        },
        "closed_loop_design": {
            "step_1": "Split 2023 data by date so future days do not leak into training.",
            "step_2": "Extract pairwise features from pass points, trajectory images, optional temperature, and optional truck purpose.",
            "step_3": "Learn feature weights from feedback when feedback labels are available.",
            "step_4": "Score validation and test periods with learned weights.",
            "step_5": "Append new feedback and rerun to correct truck correlation over time.",
        },
        "feature_meaning": {
            "pass_score": "Pearson pass correlation rescaled from [-1, 1] to [0, 1].",
            "image_similarity": "Average same-date cosine similarity of trajectory image features.",
            "purpose_match": "1 when truck purpose matches, 0 when it differs; omitted until metadata is provided.",
            "temperature_similarity": "Similarity of average temperatures observed by each truck in the split; omitted until temperature data is provided.",
        },
        "performance_goal": (
            "Closed-loop performance is evaluated by how close validation/test "
            "truck correlations stay to the training baseline. Smaller absolute "
            "delta from training means better stability."
        ),
        "learned_weights": learned_weights,
        "feedback_pairs_used": len(feedback),
        "splits": split_results,
        "performance_vs_train": _build_performance_vs_train(split_results),
    }

    if use_llm:
        run = _apply_llm_weight_update(run, llm_model, llm_provider)

    output_dir = output_root / ("vehicles_" + "_".join(str(vehicle_id) for vehicle_id in vehicle_ids))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "closed_loop_run.json"
    output_file.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    run["output_file"] = str(output_file)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop truck-correlation agent design run.")
    parser.add_argument("vehicles", nargs="+", type=int)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--passes-file", type=Path, default=DEFAULT_PASSES_FILE)
    parser.add_argument("--temperature-file", type=Path)
    parser.add_argument("--temperature-target-year", type=int)
    parser.add_argument("--metadata-file", type=Path)
    parser.add_argument("--feedback-file", type=Path)
    parser.add_argument("--train-start", default="2023-01-01")
    parser.add_argument("--train-end", default="2023-10-31")
    parser.add_argument("--validation-start", default="2023-11-01")
    parser.add_argument("--validation-end", default="2023-11-30")
    parser.add_argument("--test-start", default="2023-12-01")
    parser.add_argument("--test-end", default="2023-12-31")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-provider", choices=("anthropic", "openai"), default="anthropic")
    parser.add_argument("--llm-model")
    args = parser.parse_args()

    run = build_closed_loop_run(
        args.vehicles,
        image_root=args.image_root,
        passes_file=args.passes_file,
        temperature_file=args.temperature_file,
        temperature_target_year=args.temperature_target_year,
        metadata_file=args.metadata_file,
        feedback_file=args.feedback_file,
        train_start=args.train_start,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_start=args.test_start,
        test_end=args.test_end,
        output_root=args.output_root,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
    )

    print(f"Wrote closed-loop run to {run['output_file']}")
    print(f"Learned weights: {run['learned_weights']}")
    for split_name, split in run["splits"].items():
        print(f"{split_name}: {split['start_date']} to {split['end_date']}")
        for row in split["pairwise_relationships"]:
            score = row["learned_similarity"]
            score_text = "n/a" if score is None else f"{score:.3f}"
            print(
                f"  Vehicle {row['vehicle_a']} + {row['vehicle_b']}: "
                f"score={score_text}, {row['scenario_label']}"
            )
    print("performance vs training:")
    for split_name, performance in run["performance_vs_train"].items():
        delta = performance["mean_absolute_delta_from_train"]
        delta_text = "n/a" if delta is None else f"{delta:.3f}"
        print(f"  {split_name}: mean absolute delta from train={delta_text}")
    if "llm_weight_update" in run:
        print(f"LLM updated weights: {run['llm_weight_update']['updated_weights']}")
        print("LLM performance vs training:")
        for split_name, performance in run["llm_performance_vs_train"].items():
            delta = performance["mean_absolute_delta_from_train"]
            delta_text = "n/a" if delta is None else f"{delta:.3f}"
            print(f"  {split_name}: mean absolute delta from train={delta_text}")


if __name__ == "__main__":
    main()
