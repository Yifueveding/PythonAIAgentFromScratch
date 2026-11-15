"""Route clusters refined with daily distance, used as planned km.

Clusters are fitted per truck on Jan-Aug days using
    [route-image PCA features (unit norm), w * standardised log(1 + GPS km)]
so routes that look alike but differ in length (e.g. extra laps) split apart.

Distance is unknown on the day being planned, so a day is assigned to a
cluster by its route image only (nearest centroid on the image dimensions);
planned km = that cluster's typical km (centroid km, back-transformed). Train
and test days use the same image-only assignment, so the model never sees the
actual km. An "oracle" assignment that also uses the actual km is reported as
an upper bound on cluster quality, not as a usable forecast.

Image features are cached in route_image_features.npz after the first run.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from image_cluster import (DEFAULT_IMAGE_SIZE, DEFAULT_PCA_COMPONENTS,  # noqa: E402
                           ImageRecord, _load_pixels, _route_ink_features)
from predict_energy import build_dataset  # noqa: E402
from predict_energy_route_cluster import _pca_fit, _pca_transform, fit_with_planned  # noqa: E402
from predict_energy_scenarios import metrics, run  # noqa: E402

SPLIT = "2023-09-01"
TRUCK = 155
FEATURE_CACHE = HERE / "route_image_features.npz"
WEIGHTS = [0.0, 0.5, 1.0, 2.0]   # 0 = image-only clusters (as before, but k-means++)
CLUSTER_COUNTS = [10, 20]
SEED = 0


def image_features(d: pd.DataFrame) -> dict:
    """Per truck: dates and PCA image features (PCA fitted on Jan-Aug)."""
    if FEATURE_CACHE.exists():
        z = np.load(FEATURE_CACHE, allow_pickle=True)
        return {int(k): (pd.to_datetime(v[0]), v[1]) for k, v in z["data"].item().items()}
    out = {}
    trucks = sorted(d["TruckId"].unique())
    for i, t in enumerate(trucks, 1):
        vdir = ROOT / "Vehicle" / f"Vehicle_{t}"
        dates = [x for x in sorted(d.loc[d["TruckId"] == t, "date"]) if (vdir / f"{x:%Y-%m-%d}.png").exists()]
        is_tr = np.array([x < pd.Timestamp(SPLIT) for x in dates])
        if is_tr.sum() < 3:
            continue
        recs = [ImageRecord(f"{x:%Y-%m-%d}", str(vdir / f"{x:%Y-%m-%d}.png")) for x in dates]
        feats = _route_ink_features(_load_pixels(recs, DEFAULT_IMAGE_SIZE))
        mean, vt = _pca_fit(feats[is_tr], min(DEFAULT_PCA_COMPONENTS, is_tr.sum() - 1))
        out[int(t)] = (pd.to_datetime(dates), _pca_transform(feats, mean, vt).astype(np.float32))
        if i % 20 == 0:
            print(f"  image features {i}/{len(trucks)} trucks", flush=True)
    np.savez_compressed(FEATURE_CACHE, data=np.array(
        {k: (np.array([x.strftime("%Y-%m-%d") for x in v[0]]), v[1]) for k, v in out.items()}, dtype=object))
    return out


def planned_km(d: pd.DataFrame, feats: dict, weight: float, k: int) -> pd.DataFrame:
    """Planned km per truck-day from distance-refined route clusters."""
    km = d.set_index(["TruckId", "date"])["truck_km"]
    rows = []
    for t, (dates, z) in feats.items():
        y = np.log1p(km.reindex(pd.MultiIndex.from_arrays([[t] * len(dates), dates])).to_numpy())
        ok = np.isfinite(y)
        dates, z, y = dates[ok], z[ok], y[ok]
        is_tr = dates < pd.Timestamp(SPLIT)
        mu, sd = y[is_tr].mean(), y[is_tr].std() + 1e-6
        ys = (y - mu) / sd
        full = np.hstack([z, weight * ys[:, None]])
        kk = min(k, is_tr.sum())
        km_fit = KMeans(n_clusters=kk, n_init=10, random_state=SEED).fit(full[is_tr])
        cent = km_fit.cluster_centers_
        # cluster typical km = mean actual km of its training members
        typical = np.array([np.expm1(y[is_tr][km_fit.labels_ == c]).mean() for c in range(kk)])
        img_lab = np.linalg.norm(z[:, None, :] - cent[None, :, :-1], axis=2).argmin(axis=1)
        orc_lab = np.linalg.norm(full[:, None, :] - cent[None, :, :], axis=2).argmin(axis=1)
        rows.append(pd.DataFrame({"TruckId": t, "date": dates,
                                  "planned_km": typical[img_lab], "oracle_km": typical[orc_lab]}))
    return d.merge(pd.concat(rows), on=["TruckId", "date"], how="inner")


def km_quality(test: pd.DataFrame, col: str) -> tuple[float, float]:
    r = np.corrcoef(test[col], test["truck_km"])[0, 1]
    mape = np.median(np.abs(test[col] - test["truck_km"]) / test["truck_km"].clip(lower=1))
    return r, mape


def main():
    d = build_dataset()
    print("loading route image features (cached after first run)...")
    feats = image_features(d)

    # --- 1. how well does each clustering estimate km? (cheap, no model fit) ---
    grid = []
    for k in CLUSTER_COUNTS:
        for w in WEIGHTS:
            dd = planned_km(d, feats, w, k)
            te = dd[dd["date"] >= SPLIT]
            r, mape = km_quality(te, "planned_km")
            ro, mapeo = km_quality(te, "oracle_km")
            grid.append(dict(k=k, weight=w, r=r, mape=mape, oracle_r=ro, oracle_mape=mapeo))
            print(f"k={k:2d} w={w:.1f}: image-only r={r:.3f} med.err={mape:.1%} | oracle r={ro:.3f} med.err={mapeo:.1%}")
    grid = pd.DataFrame(grid)
    best = grid.sort_values("r", ascending=False).iloc[0]
    print(f"best image-only setting: k={int(best.k)}, w={best.weight}")

    # --- 2. energy model with the best refined clusters ----------------------
    dd = planned_km(d, feats, best.weight, int(best.k))
    train, test = dd[dd["date"] < SPLIT], dd[dd["date"] >= SPLIT]
    results = {
        "No distance": run(train, test, None, quantiles=True),
        "Refined cluster km": fit_with_planned(train, test),
        "Oracle cluster km\n(uses actual km)": fit_with_planned(
            train.assign(planned_km=train["oracle_km"]), test.assign(planned_km=test["oracle_km"])),
        "Actual km": run(train, test, 0.0, quantiles=True),
    }
    scores = {k: metrics(test, p) for k, p in results.items()}
    for k, s in scores.items():
        print(k.replace("\n", " ") + ": " + ", ".join(f"{m} {v:.3f}" for m, v in s.items()))

    # --- figure -----------------------------------------------------------------
    fig = plt.figure(figsize=(16, 11), dpi=170)
    gs = fig.add_gridspec(2, 2)
    ax_g, ax_b, ax_ts = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    for k, ls in zip(CLUSTER_COUNTS, ["-", "--"]):
        g = grid[grid["k"] == k]
        ax_g.plot(g["weight"], g["r"], color="#7e5fa6", ls=ls, marker="o", lw=2, label=f"Image-only assignment, k={k}")
        ax_g.plot(g["weight"], g["oracle_r"], color="#9aa0a6", ls=ls, marker="s", lw=1.5, label=f"Oracle (uses actual km), k={k}")
    ax_g.set_xlabel("Weight on daily distance when fitting clusters (0 = image only)")
    ax_g.set_ylabel("Correlation of cluster km with actual km (Sep–Oct)")
    ax_g.set_ylim(0, 1)
    ax_g.set_title("(a) Does refining clusters with distance improve planned km?", fontsize=11)
    ax_g.legend(frameon=False, fontsize=9, loc="lower right")
    ax_g.grid(color="#e5e5e5")
    ax_g.set_axisbelow(True)

    names = list(scores)
    colors = ["#b8434f", "#7e5fa6", "#9aa0a6", "#2a6ebb"]
    x = np.arange(len(names))
    bars = ax_b.bar(x, [scores[n]["R2"] for n in names], color=colors, width=0.6)
    for b, n in zip(bars, names):
        s = scores[n]
        ax_b.text(b.get_x() + b.get_width() / 2, max(b.get_height(), 0) + 0.02,
                  f"R² {s['R2']:.2f}\nMAE {s['MAE']:.1f}\ncover {s['coverage']:.0%}", ha="center", va="bottom", fontsize=9)
    ax_b.set_xticks(x, names, fontsize=9)
    ax_b.set_ylim(0, 1.25)
    ax_b.set_ylabel("R² (truck-day, Sep–Oct)")
    ax_b.set_title(f"(b) Energy accuracy (refined clusters: k={int(best.k)}, weight={best.weight})", fontsize=11)
    ax_b.grid(axis="y", color="#e5e5e5")
    ax_b.set_axisbelow(True)

    mask = (test["TruckId"] == TRUCK).to_numpy()
    tt = test[mask]
    p = results["Refined cluster km"]
    y, pred, lo, hi = tt["energy"].to_numpy(), p["pred"][mask], p["lo"][mask], p["hi"][mask]
    cov = np.mean((y >= lo) & (y <= hi))
    ax_ts.fill_between(tt["date"], lo, hi, color="#7e5fa6", alpha=0.18, lw=0, label="90% interval")
    ax_ts.plot(tt["date"], y, color="black", lw=1.2, marker="o", ms=2.5, label="Actual")
    ax_ts.plot(tt["date"], pred, color="#7e5fa6", lw=1.8, label="Predicted (refined cluster km)")
    ax_ts.set_title(f"(c) Truck {TRUCK}, Sep–Oct: R² {r2_score(y, pred):.2f}, MAE {mean_absolute_error(y, pred):.1f}, "
                    f"90% coverage {cov:.0%}", fontsize=11)
    ax_ts.set_ylabel("Daily energy")
    ax_ts.set_ylim(0, max(y.max(), hi.max()) * 1.1)
    ax_ts.set_xlim(tt["date"].min(), tt["date"].max())
    ax_ts.legend(frameon=False, loc="upper left", ncol=3)
    ax_ts.grid(color="#e5e5e5")
    ax_ts.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(HERE / "energy_prediction_route_cluster_refined.png")
    plt.close(fig)
    print(f"truck {TRUCK}: R2 {r2_score(y, pred):.3f}, MAE {mean_absolute_error(y, pred):.2f}, coverage {cov:.0%}")


if __name__ == "__main__":
    main()
