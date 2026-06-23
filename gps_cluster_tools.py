import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from langchain.tools import Tool
from PIL import Image


VEHICLE_NUMBER = 1994
IMAGE_SIZE = 32
PCA_COMPONENTS = 21
N_CLUSTERS = 10
IMAGE_ROOT = Path(f"Vehicle_{VEHICLE_NUMBER}")
CLUSTER_INDEX_PATH = Path(f"gps_cluster_index_{VEHICLE_NUMBER}.csv")

_CLUSTER_MODEL = None


def _image_to_feature(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0


def _load_image_features() -> tuple[list[str], np.ndarray]:
    image_paths = sorted(IMAGE_ROOT.glob("*.png"))
    dates = [path.stem for path in image_paths]
    features = np.vstack([_image_to_feature(path) for path in image_paths])
    return dates, features


def _fit_pca(features: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    centered = features - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    transformed = centered @ components.T
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    transformed = transformed / np.maximum(norms, 1e-8)
    return mean, components, transformed


def _fit_kmeans(features: np.ndarray, n_clusters: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    initial_indices = np.linspace(0, len(features) - 1, n_clusters, dtype=int)
    centroids = features[initial_indices].copy()

    for _ in range(max_iter):
        distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        new_centroids = centroids.copy()

        for cluster_id in range(n_clusters):
            members = features[labels == cluster_id]
            if len(members):
                new_centroids[cluster_id] = members.mean(axis=0)

        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
    labels = distances.argmin(axis=1)
    return labels, centroids


def _write_cluster_index(dates: list[str], labels: np.ndarray, distances: np.ndarray) -> None:
    with CLUSTER_INDEX_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "vehicle_number", "cluster", "distance_to_centroid"])
        for date, label, distance in zip(dates, labels, distances):
            writer.writerow([date, VEHICLE_NUMBER, int(label), float(distance)])


def _build_cluster_model() -> dict:
    dates, raw_features = _load_image_features()
    if len(dates) < N_CLUSTERS:
        raise ValueError(f"Need at least {N_CLUSTERS} images in {IMAGE_ROOT} to cluster trajectories.")

    n_components = min(PCA_COMPONENTS, len(dates) - 1, raw_features.shape[1])
    mean, components, features = _fit_pca(raw_features, n_components)
    labels, centroids = _fit_kmeans(features, N_CLUSTERS)
    distances = np.linalg.norm(features - centroids[labels], axis=1)
    _write_cluster_index(dates, labels, distances)

    return {
        "dates": dates,
        "mean": mean,
        "components": components,
        "features": features,
        "labels": labels,
        "centroids": centroids,
        "distances": distances,
    }


def _get_cluster_model() -> dict:
    global _CLUSTER_MODEL
    if _CLUSTER_MODEL is None:
        _CLUSTER_MODEL = _build_cluster_model()
    return _CLUSTER_MODEL


def _extract_date(query: str) -> Optional[str]:
    match = re.search(r"\d{4}-\d{2}-\d{2}", query)
    return match.group(0) if match else None


def search_gps_cluster(query: str) -> str:
    """Look up the image cluster for a GPS trajectory date."""
    model = _get_cluster_model()
    date = _extract_date(query)
    if date is None:
        return json.dumps(
            {
                "status": "missing_date",
                "message": "Please include a date in YYYY-MM-DD format.",
                "vehicle_number": VEHICLE_NUMBER,
            }
        )

    image_path = IMAGE_ROOT / f"{date}.png"
    if not image_path.exists():
        return json.dumps(
            {
                "status": "image_not_found",
                "date": date,
                "vehicle_number": VEHICLE_NUMBER,
                "message": f"No GPS trajectory image exists at {image_path}.",
            }
        )

    feature = _image_to_feature(image_path)
    transformed = (feature - model["mean"]) @ model["components"].T
    transformed = transformed / max(float(np.linalg.norm(transformed)), 1e-8)
    distances = np.linalg.norm(model["centroids"] - transformed, axis=1)
    cluster = int(distances.argmin())
    nearest_indices = np.argsort(np.linalg.norm(model["features"] - transformed, axis=1))[:5]

    return json.dumps(
        {
            "status": "ok",
            "date": date,
            "vehicle_number": VEHICLE_NUMBER,
            "cluster": cluster,
            "distance_to_centroid": float(distances[cluster]),
            "similar_dates": [model["dates"][i] for i in nearest_indices if model["dates"][i] != date][:3],
            "cluster_index_file": str(CLUSTER_INDEX_PATH),
        }
    )


def save_gps_result(data: str, filename: str = "gps_cluster_results.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- GPS Cluster Result ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as file:
        file.write(formatted_text)

    return f"GPS cluster result saved to {filename}"


search_tool = Tool(
    name="search",
    func=search_gps_cluster,
    description="Query a GPS trajectory date in YYYY-MM-DD format and return its image cluster.",
)

save_tool = Tool(
    name="save_text_to_file",
    func=save_gps_result,
    description="Save the GPS trajectory cluster result to a text file.",
)
