"""Energy prediction with planned km taken from the truck's route cluster.

Assumes the day's route is known in advance but its driven distance is not.
For each truck, the daily route images (Vehicle/Vehicle_<id>/<date>.png) are
clustered with the same method as image_cluster.py (route-ink pixels -> PCA
(21) -> k-means (10)), fitted on Jan-Aug only. Sep-Oct days are assigned to
the nearest Jan-Aug cluster. Planned km for a day = mean GPS km of the other
training days in its cluster (leave-one-out for training days, so a day never
sees its own km).

The model is the same as predict_energy_scenarios.py with ``planned_km``
replaced by this route-cluster estimate. Trained Jan-Aug, tested Sep-Oct.
Cluster assignments are cached in route_cluster_planned_km.csv.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from image_cluster import (DEFAULT_CLUSTERS, DEFAULT_IMAGE_SIZE, DEFAULT_PCA_COMPONENTS,  # noqa: E402
                           _load_pixels, _route_ink_features, ImageRecord)
from multi_truck_scenario import _fit_kmeans  # noqa: E402
from predict_energy import GB_PARAMS, build_dataset  # noqa: E402
from predict_energy_scenarios import BASE, design, metrics, run  # noqa: E402

SPLIT = "2023-09-01"
TRUCK = 155
CACHE = HERE / "route_cluster_planned_km.csv"


def _pca_fit(features: np.ndarray, n_components: int):
    mean = features.mean(axis=0)
    _, _, vt = np.linalg.svd(features - mean, full_matrices=False)
    return mean, vt[:n_components]


def _pca_transform(features, mean, vt):
    z = (features - mean) @ vt.T
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)


def cluster_truck(truck_id: int, days: pd.DataFrame) -> pd.DataFrame:
    """Route cluster per day for one truck, fitted on training days only."""
    vdir = ROOT / "Vehicle" / f"Vehicle_{truck_id}"
    days = days[days["date"].map(lambda d: (vdir / f"{d:%Y-%m-%d}.png").exists())]
    train = days[days["date"] < SPLIT]
    if len(train) < 3:
        return pd.DataFrame()
    records = [ImageRecord(f"{d:%Y-%m-%d}", str(vdir / f"{d:%Y-%m-%d}.png")) for d in days["date"]]
    feats = _route_ink_features(_load_pixels(records, DEFAULT_IMAGE_SIZE))
    is_tr = (days["date"] < SPLIT).to_numpy()
    k = min(DEFAULT_CLUSTERS, is_tr.sum())
    mean, vt = _pca_fit(feats[is_tr], min(DEFAULT_PCA_COMPONENTS, is_tr.sum() - 1))
    z = _pca_transform(feats, mean, vt)
    tr_labels, centroids = _fit_kmeans(z[is_tr], k)
    labels = np.linalg.norm(z[:, None, :] - centroids[None, :, :], axis=2).argmin(axis=1)
    labels[is_tr] = tr_labels
    return days.assign(cluster=labels)[["TruckId", "date", "cluster"]]


def route_clusters(d: pd.DataFrame) -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, parse_dates=["date"])
    parts = []
    trucks = sorted(d["TruckId"].unique())
    for i, t in enumerate(trucks, 1):
        parts.append(cluster_truck(t, d[d["TruckId"] == t][["TruckId", "date"]]))
        if i % 20 == 0:
            print(f"  clustered {i}/{len(trucks)} trucks", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(CACHE, index=False)
    return out


def add_planned_km(d: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    d = d.merge(clusters, on=["TruckId", "date"], how="inner")
    is_tr = d["date"] < SPLIT
    grp = d[is_tr].groupby(["TruckId", "cluster"])["truck_km"].agg(["sum", "count"])
    d = d.join(grp, on=["TruckId", "cluster"])
    loo = (d["sum"] - d["truck_km"]) / (d["count"] - 1)
    full = d["sum"] / d["count"]
    d["planned_km"] = np.where(is_tr, loo, full)
    # Singleton clusters (or test days in a cluster with no other training day):
    # fall back to the truck's own training-period mean km.
    truck_mean = d[is_tr].groupby("TruckId")["truck_km"].mean()
    d["planned_km"] = d["planned_km"].where(np.isfinite(d["planned_km"]), d["TruckId"].map(truck_mean))
    return d.drop(columns=["sum", "count"]).dropna(subset=["planned_km"])


def fit_with_planned(train, test, quantiles=True):
    numeric = BASE + ["planned_km"]
    Xtr = design(train, numeric)
    Xte = design(test, numeric, Xtr.columns)
    out = {"pred": GradientBoostingRegressor(**GB_PARAMS).fit(Xtr, train["energy"]).predict(Xte)}
    if quantiles:
        for k, a in [("lo", 0.05), ("hi", 0.95)]:
            out[k] = GradientBoostingRegressor(loss="quantile", alpha=a, **GB_PARAMS).fit(Xtr, train["energy"]).predict(Xte)
    return out


def main():
    d = build_dataset()
    print("clustering route images (cached after first run)...")
    d = add_planned_km(d, route_clusters(d))
    train, test = d[d["date"] < SPLIT], d[d["date"] >= SPLIT]
    print(f"{len(d)} truck-days with route images ({len(train)} train / {len(test)} test), {d['TruckId'].nunique()} trucks")

    # How good is the route-cluster km as a distance estimate?
    km_err = np.log(test["planned_km"].clip(lower=0.1) / test["truck_km"].clip(lower=0.1))
    km_r = np.corrcoef(test["planned_km"], test["truck_km"])[0, 1]
    km_mape = np.median(np.abs(test["planned_km"] - test["truck_km"]) / test["truck_km"].clip(lower=1))
    print(f"planned vs actual km (test): r={km_r:.3f}, median abs % error={km_mape:.1%}, "
          f"log-error sd={km_err.std():.2f}")

    results = {
        "No distance": run(train, test, None, quantiles=True),
        "Route-cluster km": fit_with_planned(train, test),
        "Actual km": run(train, test, 0.0, quantiles=True),
    }
    scores = {k: metrics(test, p) for k, p in results.items()}
    for k, s in scores.items():
        print(f"{k}: " + ", ".join(f"{m} {v:.3f}" for m, v in s.items()))

    # --- figure ---------------------------------------------------------------
    fig = plt.figure(figsize=(16, 11), dpi=170)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_km, ax_bar, ax_ts = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    lim = np.percentile(np.r_[test["truck_km"], test["planned_km"]], 99.5)
    ax_km.scatter(test["truck_km"], test["planned_km"], s=4, alpha=0.25, color="#2a6ebb", lw=0)
    ax_km.plot([0, lim], [0, lim], color="black", lw=1, ls="--")
    ax_km.set_xlim(0, lim)
    ax_km.set_ylim(0, lim)
    ax_km.set_xlabel("Actual GPS km that day")
    ax_km.set_ylabel("Planned km (route-cluster typical)")
    ax_km.set_title(f"(a) Route-cluster planned km vs actual, Sep–Oct\nr = {km_r:.2f}, "
                    f"median abs. error {km_mape:.0%}", fontsize=11)
    ax_km.grid(color="#e5e5e5")
    ax_km.set_axisbelow(True)

    names = list(scores)
    colors = ["#b8434f", "#7e5fa6", "#2a6ebb"]
    x = np.arange(len(names))
    r2 = [scores[n]["R2"] for n in names]
    bars = ax_bar.bar(x, r2, color=colors, width=0.6)
    for b, n in zip(bars, names):
        s = scores[n]
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"R² {s['R2']:.2f}\nMAE {s['MAE']:.1f}\ncover {s['coverage']:.0%}",
                    ha="center", va="bottom", fontsize=9)
    ax_bar.set_xticks(x, names)
    ax_bar.set_ylim(0, 1.25)
    ax_bar.set_ylabel("R² (truck-day, Sep–Oct)")
    ax_bar.set_title("(b) Accuracy by distance input — all trucks", fontsize=11)
    ax_bar.grid(axis="y", color="#e5e5e5")
    ax_bar.set_axisbelow(True)

    mask = (test["TruckId"] == TRUCK).to_numpy()
    tt = test[mask]
    p = results["Route-cluster km"]
    y, pred, lo, hi = tt["energy"].to_numpy(), p["pred"][mask], p["lo"][mask], p["hi"][mask]
    cov = np.mean((y >= lo) & (y <= hi))
    ax_ts.fill_between(tt["date"], lo, hi, color="#7e5fa6", alpha=0.18, lw=0, label="90% interval")
    ax_ts.plot(tt["date"], y, color="black", lw=1.2, marker="o", ms=2.5, label="Actual")
    ax_ts.plot(tt["date"], pred, color="#7e5fa6", lw=1.8, label="Predicted (route-cluster km)")
    ax_ts.set_title(f"(c) Truck {TRUCK}, Sep–Oct: R² {r2_score(y, pred):.2f}, MAE {mean_absolute_error(y, pred):.1f}, "
                    f"90% coverage {cov:.0%}", fontsize=11)
    ax_ts.set_ylabel("Daily energy")
    ax_ts.set_ylim(0, max(y.max(), hi.max()) * 1.1)
    ax_ts.set_xlim(tt["date"].min(), tt["date"].max())
    ax_ts.legend(frameon=False, loc="upper left", ncol=3)
    ax_ts.grid(color="#e5e5e5")
    ax_ts.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(HERE / "energy_prediction_route_cluster.png")
    plt.close(fig)
    print(f"truck {TRUCK}: R2 {r2_score(y, pred):.3f}, MAE {mean_absolute_error(y, pred):.2f}, coverage {cov:.0%}")


if __name__ == "__main__":
    main()
