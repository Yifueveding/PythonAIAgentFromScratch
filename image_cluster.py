import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from multi_truck_scenario import _fit_kmeans, _fit_pca


DEFAULT_IMAGE_ROOT = Path("Vehicle")
DEFAULT_OUTPUT_DIR = Path("image_cluster")
DEFAULT_IMAGE_SIZE = 128
DEFAULT_LATENT_DIM = 64
DEFAULT_PCA_COMPONENTS = 21
DEFAULT_CLUSTERS = 10


@dataclass(frozen=True)
class ImageRecord:
    date: str
    image_path: str


def _normalize_date(value: str) -> str:
    for date_format in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value[:10], date_format).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value[:10]


def _in_period(date: str, start_date: str, end_date: str) -> bool:
    return start_date <= date <= end_date


def _vehicle_image_dir(image_root: Path, vehicle_id: int) -> Path:
    nested = image_root / f"Vehicle_{vehicle_id}"
    if nested.exists():
        return nested
    return Path(f"Vehicle_{vehicle_id}")


def _load_image_records(
    image_root: Path,
    vehicle_id: int,
    start_date: str,
    end_date: str,
) -> list[ImageRecord]:
    vehicle_dir = _vehicle_image_dir(image_root, vehicle_id)
    if not vehicle_dir.exists():
        raise ValueError(f"No image directory found for Vehicle {vehicle_id}: {vehicle_dir}")

    records = []
    for path in sorted(vehicle_dir.glob("*.png")):
        date = _normalize_date(path.stem)
        if _in_period(date, start_date, end_date):
            records.append(ImageRecord(date=date, image_path=str(path)))
    return records


def _load_pixels(records: list[ImageRecord], image_size: int) -> np.ndarray:
    images = []
    for record in records:
        image = Image.open(record.image_path).convert("L").resize((image_size, image_size))
        images.append(np.asarray(image, dtype=np.float32) / 255.0)
    if not images:
        return np.empty((0, image_size, image_size), dtype=np.float32)
    return np.stack(images)


def _route_ink_features(pixels: np.ndarray) -> np.ndarray:
    features = (1.0 - pixels).reshape((len(pixels), -1))
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-8)


def _build_autoencoder(image_size: int, latent_dim: int):
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model
    except ImportError as e:
        raise RuntimeError(
            "TensorFlow is required for --feature-method autoencoder. "
            "Install tensorflow or use --feature-method pca."
        ) from e

    class Autoencoder(Model):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = tf.keras.Sequential(
                [
                    layers.Input(shape=(image_size, image_size, 1)),
                    layers.BatchNormalization(),
                    layers.Conv2D(32, (3, 3), activation="relu", padding="same", strides=2),
                    layers.BatchNormalization(),
                    layers.Conv2D(64, (3, 3), activation="relu", padding="same", strides=2),
                    layers.Flatten(),
                    layers.Dense(latent_dim, activation="relu"),
                ]
            )
            reduced_size = image_size // 4
            self.decoder = tf.keras.Sequential(
                [
                    layers.Input(shape=(latent_dim,)),
                    layers.Dense(reduced_size * reduced_size * 64, activation="relu"),
                    layers.Reshape((reduced_size, reduced_size, 64)),
                    layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same", strides=2),
                    layers.BatchNormalization(),
                    layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same", strides=2),
                    layers.BatchNormalization(),
                    layers.Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same", strides=1),
                ]
            )

        def call(self, x):
            encoded = self.encoder(x)
            return self.decoder(encoded)

        def masked_mse_loss(self, y_true, y_pred):
            mask = tf.where(y_true < 0.98, 1.0, 0.1)
            mask = tf.cast(mask, dtype=tf.float32)
            loss = tf.square(y_true - y_pred) * mask
            return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)

    return Autoencoder()


def _autoencoder_features(
    pixels: np.ndarray,
    image_size: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    if len(pixels) < 2:
        return _route_ink_features(pixels), {"feature_method": "pca", "reason": "fewer_than_two_images"}

    model = _build_autoencoder(image_size, latent_dim)
    x_train = pixels[..., None].astype("float32")

    import tensorflow as tf

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=3,
        restore_best_weights=True,
    )
    model.compile(optimizer="adam", loss=model.masked_mse_loss)
    history = model.fit(
        x_train,
        x_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0,
    )
    latent = model.encoder(x_train).numpy()
    return latent, {
        "feature_method": "autoencoder",
        "latent_dim": latent_dim,
        "epochs_requested": epochs,
        "epochs_trained": len(history.history.get("loss", [])),
        "final_loss": float(history.history["loss"][-1]) if history.history.get("loss") else None,
    }


def _cluster_features(
    features: np.ndarray,
    requested_clusters: int,
    pca_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(features) == 0:
        return np.asarray([], dtype=int), np.empty((0, 0)), np.empty((0, 0))
    cluster_count = min(max(1, requested_clusters), len(features))
    if cluster_count == 1:
        return np.zeros(len(features), dtype=int), features, np.asarray([features.mean(axis=0)])

    n_components = min(pca_components, len(features) - 1, features.shape[1])
    reduced = _fit_pca(features, n_components)
    labels, centroids = _fit_kmeans(reduced, cluster_count)
    return labels, reduced, centroids


def _cluster_summary(records: list[ImageRecord], labels: np.ndarray, reduced: np.ndarray, centroids: np.ndarray) -> list[dict]:
    counts = Counter(int(label) for label in labels)
    summaries = []
    for cluster_id, days in counts.most_common():
        indices = [index for index, label in enumerate(labels) if int(label) == cluster_id]
        distances = np.linalg.norm(reduced[indices] - centroids[cluster_id], axis=1) if len(indices) else []
        representative_index = indices[int(np.argmin(distances))] if len(indices) else None
        summaries.append(
            {
                "cluster": cluster_id,
                "days": days,
                "share": days / len(records) if records else None,
                "representative_date": records[representative_index].date if representative_index is not None else None,
                "example_dates": [records[index].date for index in indices[:10]],
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["date", "vehicle_id", "image_path", "cluster", "distance_to_centroid"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_cluster_contact_sheet(
    path: Path,
    rows: list[dict],
    max_images_per_cluster: int = 12,
    thumb_size: int = 96,
) -> None:
    clusters = {}
    for row in rows:
        clusters.setdefault(row["cluster"], []).append(row)
    if not clusters:
        return

    cluster_ids = sorted(clusters)
    label_height = 24
    width = max_images_per_cluster * thumb_size
    height = len(cluster_ids) * (thumb_size + label_height)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for row_index, cluster_id in enumerate(cluster_ids):
        y = row_index * (thumb_size + label_height)
        draw.text((4, y + 4), f"Cluster {cluster_id}", fill="black")
        for col_index, row in enumerate(clusters[cluster_id][:max_images_per_cluster]):
            image = Image.open(row["image_path"]).convert("RGB").resize((thumb_size, thumb_size))
            x = col_index * thumb_size
            canvas.paste(image, (x, y + label_height))
    canvas.save(path)


def cluster_vehicle_images(
    vehicle_id: int,
    start_date: str,
    end_date: str,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    feature_method: str = "pca",
    n_clusters: int = DEFAULT_CLUSTERS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    latent_dim: int = DEFAULT_LATENT_DIM,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    epochs: int = 50,
    batch_size: int = 8,
) -> dict:
    normalized_start = _normalize_date(start_date)
    normalized_end = _normalize_date(end_date)
    if normalized_start > normalized_end:
        raise ValueError("--start-date must be before or equal to --end-date.")

    records = _load_image_records(image_root, vehicle_id, normalized_start, normalized_end)
    if not records:
        raise ValueError(
            f"No Vehicle {vehicle_id} PNG images found from {normalized_start} to {normalized_end}."
        )

    pixels = _load_pixels(records, image_size)
    method_details = {"feature_method": feature_method}
    if feature_method == "autoencoder":
        features, method_details = _autoencoder_features(
            pixels,
            image_size,
            latent_dim,
            epochs,
            batch_size,
        )
    elif feature_method == "pca":
        features = _route_ink_features(pixels)
        method_details = {"feature_method": "pca", "source": "route_ink_pixels"}
    else:
        raise ValueError(f"Unsupported feature method: {feature_method}")

    labels, reduced, centroids = _cluster_features(features, n_clusters, pca_components)
    distances = np.linalg.norm(reduced - centroids[labels], axis=1) if len(labels) else np.asarray([])
    rows = [
        {
            "date": record.date,
            "vehicle_id": vehicle_id,
            "image_path": record.image_path,
            "cluster": int(label),
            "distance_to_centroid": float(distance),
        }
        for record, label, distance in zip(records, labels, distances)
    ]

    result = {
        "vehicle_id": vehicle_id,
        "start_date": normalized_start,
        "end_date": normalized_end,
        "image_count": len(records),
        "method": {
            **method_details,
            "image_size": image_size,
            "pca_components": min(pca_components, max(1, len(records) - 1), features.shape[1]),
            "requested_clusters": n_clusters,
            "actual_clusters": len(set(int(label) for label in labels)),
            "notebook_reference": "autoeconoder_cnn_best_example.ipynb",
        },
        "cluster_summary": _cluster_summary(records, labels, reduced, centroids),
        "clusters_by_date": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = (
        f"vehicle_{vehicle_id}_{normalized_start}_{normalized_end}_{result['method']['feature_method']}"
    )
    json_file = output_dir / f"{base_name}.json"
    csv_file = output_dir / f"{base_name}.csv"
    contact_sheet_file = output_dir / f"{base_name}_clusters.png"
    json_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_file, rows)
    _write_cluster_contact_sheet(contact_sheet_file, rows)

    result["output_files"] = {
        "json": str(json_file),
        "csv": str(csv_file),
        "cluster_contact_sheet": str(contact_sheet_file),
    }
    json_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster one truck's route images over a selected period, following the "
            "autoencoder/PCA/k-means workflow from autoeconoder_cnn_best_example.ipynb."
        )
    )
    parser.add_argument("--vehicle-id", type=int, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-method", choices=("pca", "autoencoder"), default="pca")
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    result = cluster_vehicle_images(
        vehicle_id=args.vehicle_id,
        start_date=args.start_date,
        end_date=args.end_date,
        image_root=args.image_root,
        output_dir=args.output_dir,
        feature_method=args.feature_method,
        n_clusters=args.clusters,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        pca_components=args.pca_components,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print(f"Clustered {result['image_count']} images for Vehicle {result['vehicle_id']}.")
    for summary in result["cluster_summary"]:
        print(
            f"  Cluster {summary['cluster']}: days={summary['days']}, "
            f"share={summary['share']:.1%}, representative={summary['representative_date']}"
        )
    print(f"Wrote JSON to {result['output_files']['json']}")
    print(f"Wrote CSV to {result['output_files']['csv']}")
    print(f"Wrote cluster image sheet to {result['output_files']['cluster_contact_sheet']}")


if __name__ == "__main__":
    main()
