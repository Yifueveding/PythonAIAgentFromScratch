# `main.py` Workflow Summary

`main.py` is the statistical scenario-generation entry point. It builds truck route scenarios by combining truck-day data, stop-profile correlations, image route clusters, fleet metadata, and weighted sampling.

## Main Inputs

- `--target-vehicle`: the main truck to build scenarios around.
- `--other-vehicles`: candidate trucks that may appear with the target truck.
- `--start-date`, `--end-date`: selected analysis period.
- `--image-feature-method`: route-image feature method, either `pca` or `autoencoder`.
- `--image-clusters`: number of route clusters per truck.
- `--scenario-days`: number of final scenario days to sample.
- `--seed`: random seed for repeatable sampling.
- `--validation-loop`: optional branch for weight tuning.
- `--seeds`: optional branch for running multiple scenario seeds.

## Default Scenario Workflow

The default CLI path calls `build_scenario_generation(...)`.

```text
main.py
└── main()
    └── build_scenario_generation(...)
        ├── Normalize dates
        │   ├── _normalize_date(start_date)
        │   ├── _normalize_date(end_date)
        │   └── _date_range(start_date, end_date)
        │
        ├── Build output folder
        │   └── scenario_generation/
        │       └── target_<target>_others_<others>_<start>_<end>/
        │
        ├── Step 1: Build truck-day representations
        │   ├── TruckDayLookup()
        │   ├── lookup.get(vehicle_id, date)
        │   ├── _replace_none_with_nan(...)
        │   └── Write data_representations.json
        │
        ├── Step 2: Compute stop-profile correlations
        │   ├── calculate_period_correlations(...)
        │   │   └── Reads vehicle_day_zone_passes.csv
        │   └── Write stop_profile_correlations.json
        │
        ├── Step 3: Cluster route images
        │   ├── For each vehicle:
        │   │   └── cluster_vehicle_images(...)
        │   └── Save per-truck image cluster outputs
        │
        ├── Step 4: Compute statistical argument correlations
        │   ├── _normalize_weights(...)
        │   ├── calculate_statistical_argument_correlation(...)
        │   │   ├── stop_profile_correlation
        │   │   ├── route_similarity
        │   │   ├── purpose_match
        │   │   ├── age_similarity
        │   │   ├── kms_similarity
        │   │   └── duty_cycle_match
        │   └── Write argument_correlations.json
        │
        ├── Step 5: Convert correlations into appearance frequencies
        │   ├── _target_pair_rows(...)
        │   └── _truck_appearance_frequencies(...)
        │       └── Positive target-to-other correlations are normalized
        │
        ├── Step 6: Convert route clusters into route frequencies
        │   ├── _target_route_rows(...)
        │   │   └── Target routes weighted by route_appearance_frequency
        │   └── _route_appearance_rows(...)
        │       └── Other-truck routes weighted by joint_scenario_frequency
        │
        ├── Step 7: Sample final scenario days
        │   └── _sample_final_scenario_days(...)
        │       ├── Sample target route by route_appearance_frequency
        │       └── Sample other truck route by joint_scenario_frequency
        │
        └── Write final outputs
            ├── image_clusters.json
            ├── appearance_frequencies.json
            ├── final_two_day_scenario.json
            └── scenario_summary.json
```

## Key Calculations

### Truck Appearance Frequency

For each other truck, `main.py` takes the statistical argument correlation between the target truck and that other truck.

```text
raw_score = max(0, total_correlation)
appearance_frequency = raw_score / sum(all positive raw_scores)
```

If all raw scores are zero, every other truck gets `appearance_frequency = 0.0`.

### Route Appearance Frequency

For each route cluster:

```text
route_appearance_frequency = cluster_days / period_days
```

### Joint Scenario Frequency

For routes belonging to other trucks:

```text
joint_scenario_frequency =
    truck_appearance_frequency * route_appearance_frequency
```

The final scenario samples:

- target truck route by `route_appearance_frequency`
- other truck route by `joint_scenario_frequency`

## CLI Branch Tree

```text
main()
├── If --validation-loop is provided
│   └── build_validation_loop(...)
│       ├── Pick validation dates
│       ├── Cache image clusters once
│       │   └── _build_image_cluster_cache(...)
│       ├── For each iteration
│       │   ├── Generate nearby weight candidates
│       │   │   └── _neighbor_weights(...)
│       │   ├── For each candidate
│       │   │   ├── For each scenario seed
│       │   │   │   ├── build_scenario_generation(...)
│       │   │   │   ├── _validate_scenario_distances(...)
│       │   │   │   └── Write validation.json
│       │   │   ├── _aggregate_seed_validations(...)
│       │   │   └── Write candidate_validation_summary.json
│       │   ├── Select best candidate by validation error
│       │   └── Stop if threshold is reached
│       └── Write validation_loop_summary.json
│
├── Else if --seeds is provided
│   └── build_multi_seed_scenarios(...)
│       ├── For each seed
│       │   └── build_scenario_generation(...)
│       └── Write multi_seed_index.json
│
└── Else default path
    └── build_scenario_generation(...)
        └── Write one scenario run
```

## Output Files

Default run outputs:

```text
scenario_generation/
└── target_<target>_others_<others>_<start>_<end>/
    ├── data_representations.json
    ├── stop_profile_correlations.json
    ├── argument_correlations.json
    ├── image_clusters.json
    ├── appearance_frequencies.json
    ├── final_two_day_scenario.json
    ├── scenario_summary.json
    └── image_cluster/
        ├── vehicle_<id>_<start>_<end>_<method>.json
        ├── vehicle_<id>_<start>_<end>_<method>.csv
        └── vehicle_<id>_<start>_<end>_<method>_clusters.png
```

Validation-loop outputs:

```text
scenario_generation/
└── target_<target>_others_<others>_<start>_<end>_validation_loop/
    ├── cached_image_clusters.json
    ├── validation_loop_summary.json
    └── iteration_<n>/
        └── candidate_<n>/
            ├── candidate_validation_summary.json
            └── seed_<seed>/
                ├── scenario_summary.json
                ├── final_two_day_scenario.json
                └── validation.json
```

Multi-seed outputs:

```text
scenario_generation/
└── target_<target>_others_<others>_<start>_<end>_multi_seed/
    ├── multi_seed_index.json
    └── seed_<seed>/
        ├── scenario_summary.json
        └── final_two_day_scenario.json
```

## Short Interpretation

`main.py` first builds evidence for each truck, then scores how likely each other truck is to appear with the target truck, then scores how likely each route cluster is. The final scenario is sampled from those probabilities, so common routes and strongly related trucks are more likely to appear.

## Three Scenario-Generation Variants

The repo has three closely related scenario-generation paths. They share the same base evidence: truck-day representations, stop-zone profiles, route-image clusters, and fleet/route metadata. They differ in where the LLM is used.

| Variant | Main Script | Main Function | Output Root | What Uses LLM? | Final Sampling |
|---|---|---|---|---|---|
| No LLM scenario generation | `main.py` | `build_scenario_generation(...)` | `scenario_generation/` | Nothing | Statistical truck weights and statistical route-cluster frequencies |
| LLM scenario generation | `main_llm.py` | `build_llm_scenario_generation(...)` | `scenario_generation_llm/` | Pairwise truck correlation reasoning | LLM truck weights, but route clusters still sampled by observed cluster frequency |
| LLM cluster-and-sample generation | `route_cluster_sampling_llm.py` | `build_llm_route_cluster_sampling(...)` | `route_cluster_sampling_llm/` | Route-cluster sampling weights | LLM truck weights from an existing scenario summary plus LLM route-cluster weights |

### Variant 1: No LLM Scenario Generation

Script:

```text
main.py
```

Core call:

```text
build_scenario_generation(...)
```

Workflow:

```text
main.py
└── build_scenario_generation(...)
    ├── Build truck-day records
    │   └── TruckDayLookup().get(...)
    ├── Compute stop-profile correlations
    │   └── calculate_period_correlations(...)
    ├── Cluster route images
    │   └── cluster_vehicle_images(...)
    ├── Compute statistical pairwise argument correlations
    │   └── calculate_statistical_argument_correlation(...)
    ├── Convert target-to-other correlations into truck appearance frequencies
    ├── Convert image-cluster day counts into route appearance frequencies
    └── Sample final scenario
        ├── Target route weight: route_appearance_frequency
        └── Other route weight: truck_appearance_frequency * route_appearance_frequency
```

Important output files:

```text
scenario_generation/<run>/
├── scenario_summary.json
├── argument_correlations.json
├── appearance_frequencies.json
└── final_two_day_scenario.json
```

This is the baseline. Truck relationships are fully statistical, and route choices are based on observed route-cluster frequency.

### Variant 2: LLM Scenario Generation

Script:

```text
main_llm.py
```

Core call:

```text
build_llm_scenario_generation(...)
```

Supporting script:

```text
argument_correlation_llm.py
└── calculate_llm_argument_correlation(...)
```

Workflow:

```text
main_llm.py
└── build_llm_scenario_generation(...)
    ├── Build truck-day records
    │   └── TruckDayLookup().get(...)
    ├── Compute stop-profile correlations
    │   └── calculate_period_correlations(...)
    ├── Cluster route images
    │   └── cluster_vehicle_images(...)
    ├── Ask LLM to reason pairwise truck correlations
    │   └── calculate_llm_argument_correlation(...)
    ├── Convert LLM target-to-other correlations into truck appearance frequencies
    ├── Keep route-cluster weights statistical
    │   └── route_appearance_frequency = cluster_days / period_days
    └── Sample final scenario
        ├── Target route weight: route_appearance_frequency
        └── Other route weight: llm_truck_appearance_frequency * route_appearance_frequency
```

Important output files:

```text
scenario_generation_llm/<run>/
├── scenario_summary_llm.json
├── argument_correlations_llm.json
├── appearance_frequencies_llm.json
└── final_llm_scenario.json
```

This variant uses the LLM to decide how strongly trucks are related. It does not ask the LLM to choose route clusters; route clusters are still sampled from observed cluster frequencies.

### Variant 3: LLM Cluster-And-Sample Generation

Script:

```text
route_cluster_sampling_llm.py
```

Core call:

```text
build_llm_route_cluster_sampling(...)
```

Input:

```text
--scenario-summary <path_to_existing_scenario_summary_llm.json>
```

Usually this input comes from Variant 2:

```text
scenario_generation_llm/<run>/scenario_summary_llm.json
```

Workflow:

```text
route_cluster_sampling_llm.py
└── build_llm_route_cluster_sampling(...)
    ├── Read existing scenario summary
    │   └── Usually scenario_summary_llm.json
    ├── Read image_clusters.json from that summary
    ├── Load distance records
    │   └── vehicle_daily_distance_speed.csv
    ├── Build cluster options for every vehicle
    │   ├── current route weight
    │   ├── route days
    │   ├── representative date
    │   ├── example dates
    │   └── cluster distance stats
    ├── Compact prior argument-correlation context
    ├── Ask LLM for route_cluster_weight values
    │   └── _request_llm_route_weights(...)
    ├── Normalize LLM route-cluster weights per vehicle
    └── Sample final scenario
        ├── Target route weight: llm_route_cluster_weight
        └── Other route weight: truck_appearance_frequency * llm_route_cluster_weight
```

Important output files:

```text
route_cluster_sampling_llm/<run>_route_cluster_llm/
├── route_cluster_sampling_summary.json
├── llm_route_cluster_weights.json
└── final_route_cluster_llm_scenario.json
```

This variant does not recompute truck correlations from scratch. It starts from an existing scenario summary, usually the LLM scenario summary, and asks the LLM to improve the route-cluster sampling weights.

## Variant Relationship Tree

```text
Scenario generation methods
├── 1. No LLM
│   └── main.py
│       └── Statistical truck correlation
│       └── Statistical route-cluster frequency
│
├── 2. LLM truck correlation
│   └── main_llm.py
│       └── argument_correlation_llm.py
│           └── LLM pairwise truck correlation
│       └── Statistical route-cluster frequency
│
└── 3. LLM route-cluster sampling
    └── route_cluster_sampling_llm.py
        └── Reads scenario_summary_llm.json or another existing summary
        └── Uses existing truck appearance frequencies
        └── LLM route-cluster weights
```

## Practical Difference

- Use `main.py` when you want a deterministic statistical baseline.
- Use `main_llm.py` when you want the LLM to reason about which trucks belong together, but still want route choices grounded in observed cluster frequencies.
- Use `route_cluster_sampling_llm.py` when you already have a scenario summary and want the LLM to adjust which route clusters should be sampled for each selected truck.
