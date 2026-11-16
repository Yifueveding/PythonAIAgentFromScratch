"""Day-ahead energy forecast per truck: route known, distance unknown.

Only information available before the day starts is used:
  truck       : category, duty cycle, age, odometer, in-fleet flag
  calendar    : month, daily mean temperature (a weather forecast stands in)
  route       : planned km from route clusters (see below), zone passes
  history     : the truck's own energy and GPS km on previous active days
                (lag 1, mean and std of the last 7 active days, days since
                last active). Only past days are used.
The same-day fleet total distance and the truck's same-day GPS measures are
NOT used (they are only known after the day).

Route representation
  GPS rasters : fixed-extent binary route rasters from Positions/*.csv
                (build_route_rasters.py), clustered per truck with distance
                refinement (assigned by route only), plus the number of
                fine/coarse cells the route covers

Trained Jan-Aug, tested Sep-Oct.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from predict_energy import CATEGORICAL, GB_PARAMS, build_dataset
from predict_energy_route_cluster_refined import SPLIT

HERE = Path(__file__).resolve().parent
RASTERS = HERE / "route_rasters.npz"
K, W, SEED = 20, 0.5, 0
STATIC = ["age", "km_k", "in_fleet", "temp", "month", "zone_passes"]
HISTORY = ["lag1_energy", "lag1_km", "mean7_energy", "mean7_km", "std7_energy", "days_since"]
PANEL_TRUCKS = [155, 13873, 3019, 12263, 1988, 2921, 2705, 8065, 2201]


# --- history features ----------------------------------------------------------
def add_history(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["TruckId", "date"]).copy()
    g = d.groupby("TruckId")
    prev_e, prev_km = g["energy"].shift(1), g["truck_km"].shift(1)
    d["lag1_energy"], d["lag1_km"] = prev_e, prev_km
    d["mean7_energy"] = prev_e.groupby(d["TruckId"]).transform(lambda s: s.rolling(7, min_periods=1).mean())
    d["mean7_km"] = prev_km.groupby(d["TruckId"]).transform(lambda s: s.rolling(7, min_periods=1).mean())
    d["std7_energy"] = prev_e.groupby(d["TruckId"]).transform(lambda s: s.rolling(7, min_periods=2).std())
    d["days_since"] = g["date"].diff().dt.days
    return d


# --- GPS raster route clusters -----------------------------------------------
def raster_planned_km(d: pd.DataFrame) -> pd.DataFrame:
    z = np.load(RASTERS)
    nf, nc = int(z["fine_n"]) ** 2, int(z["coarse_n"]) ** 2
    key = pd.DataFrame({"TruckId": z["truck"], "date": pd.to_datetime(z["date"]), "i": np.arange(len(z["truck"]))})
    dd = d.merge(key, on=["TruckId", "date"], how="inner")
    fo, co = z["fine_off"], z["coarse_off"]
    rows = []
    for t, g in dd.groupby("TruckId"):
        idx = g["i"].to_numpy()
        X = np.zeros((len(idx), nf + nc), np.float32)
        for r, i in enumerate(idx):
            X[r, z["fine_idx"][fo[i]:fo[i + 1]]] = 1
            X[r, nf + z["coarse_idx"][co[i]:co[i + 1]]] = 1
        n_fine, n_coarse = X[:, :nf].sum(1), X[:, nf:].sum(1)
        is_tr = (g["date"] < SPLIT).to_numpy()
        if is_tr.sum() < 3:
            continue
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
        mean = Xn[is_tr].mean(0)
        _, _, vt = np.linalg.svd(Xn[is_tr] - mean, full_matrices=False)
        p = (Xn - mean) @ vt[:min(21, is_tr.sum() - 1)].T
        p /= np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
        y = np.log1p(g["truck_km"].to_numpy())
        ys = (y - y[is_tr].mean()) / (y[is_tr].std() + 1e-6)
        full = np.hstack([p, W * ys[:, None]])
        kk = min(K, is_tr.sum())
        km = KMeans(n_clusters=kk, n_init=10, random_state=SEED).fit(full[is_tr])
        typical = np.array([np.expm1(y[is_tr][km.labels_ == c]).mean() for c in range(kk)])
        lab = np.linalg.norm(p[:, None, :] - km.cluster_centers_[None, :, :-1], axis=2).argmin(1)
        rows.append(pd.DataFrame({"TruckId": t, "date": g["date"].to_numpy(), "raster_km": typical[lab],
                                  "n_fine": n_fine, "n_coarse": n_coarse}))
    return d.merge(pd.concat(rows), on=["TruckId", "date"], how="left")


# --- modelling -----------------------------------------------------------------
def design(df, numeric, columns=None):
    X = pd.get_dummies(df[numeric + CATEGORICAL], columns=CATEGORICAL, dtype=float).fillna(-1)
    return X if columns is None else X.reindex(columns=columns, fill_value=0.0)


def fit(train, test, numeric, quantiles=False):
    Xtr, ytr = design(train, numeric), train["energy"]
    Xte = design(test, numeric, Xtr.columns)
    out = {"pred": GradientBoostingRegressor(**GB_PARAMS).fit(Xtr, ytr).predict(Xte)}
    if quantiles:
        for k, a in [("lo", 0.05), ("hi", 0.95)]:
            out[k] = GradientBoostingRegressor(loss="quantile", alpha=a, **GB_PARAMS).fit(Xtr, ytr).predict(Xte)
    return out


def per_truck_r2(test, pred, min_days=20):
    t = test.assign(pred=pred)
    return pd.Series({tid: r2_score(g["energy"], g["pred"]) for tid, g in t.groupby("TruckId")
                      if len(g) >= min_days and g["energy"].var() > 0})


def main():
    d = build_dataset()
    d = add_history(d)
    d = raster_planned_km(d)
    d = d.dropna(subset=["raster_km"])                  # same truck-days for every variant
    train, test = d[d["date"] < SPLIT], d[d["date"] >= SPLIT]
    print(f"{len(train)} train / {len(test)} test truck-days, {d['TruckId'].nunique()} trucks")
    r = np.corrcoef(test["raster_km"], test["truck_km"])[0, 1]
    print(f"planned km vs actual (Sep-Oct): GPS raster clusters r={r:.3f}")

    variants = {
        "No route,\nno history": STATIC,
        "History only": STATIC + HISTORY,
        "GPS rasters": STATIC + ["raster_km", "n_fine", "n_coarse"],
        "GPS rasters\n+ history": STATIC + ["raster_km", "n_fine", "n_coarse"] + HISTORY,
    }
    res, summary = {}, []
    for name, cols in variants.items():
        res[name] = fit(train, test, cols, quantiles=(name == "GPS rasters\n+ history"))
        p = res[name]["pred"]
        pt = per_truck_r2(test, p)
        fleet = test.assign(pred=p).groupby("date")[["energy", "pred"]].sum()
        summary.append(dict(variant=name.replace("\n", " "), R2=r2_score(test["energy"], p),
                            MAE=mean_absolute_error(test["energy"], p), truck_median_R2=pt.median(),
                            share_R2_pos=(pt > 0).mean(), share_R2_05=(pt > 0.5).mean(),
                            fleet_MAPE=np.mean(np.abs(fleet["energy"] - fleet["pred"]) / fleet["energy"])))
    summary = pd.DataFrame(summary).set_index("variant")
    print(summary.round(3).to_string())
    best = "GPS rasters\n+ history"
    pt_best, pt_cur = per_truck_r2(test, res[best]["pred"]), per_truck_r2(test, res["No route,\nno history"]["pred"])

    # --- figure -------------------------------------------------------------------
    trucks = [t for t in PANEL_TRUCKS if t in set(test["TruckId"])]
    n = len(trucks)
    fig = plt.figure(figsize=(16, 3.0 * (n + 2)), dpi=150)
    gs = fig.add_gridspec(n + 2, 2)
    ax_b, ax_h = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    names = list(variants)
    x = np.arange(len(names))
    colors = ["#9aa0a6", "#7e5fa6", "#3fa3a3", "#2a6ebb"]
    w = 0.38
    b1 = ax_b.bar(x - w / 2, summary["R2"], w, color=colors, label="All truck-days R²")
    b2 = ax_b.bar(x + w / 2, summary["truck_median_R2"], w, color=colors, alpha=0.45, hatch="//", label="Median per-truck R²")
    for bars, col in [(b1, "R2"), (b2, "truck_median_R2")]:
        for b, v in zip(bars, summary[col]):
            ax_b.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax_b.set_xticks(x, names, fontsize=9)
    ax_b.set_ylim(min(0, summary["truck_median_R2"].min() - 0.05), 1)
    ax_b.set_ylabel("R² (Sep–Oct)")
    ax_b.set_title("(a) Forecast accuracy by route input (solid: all truck-days, hatched: median truck)", fontsize=11)
    ax_b.legend(frameon=False, loc="upper left", fontsize=9)
    ax_b.grid(axis="y", color="#e5e5e5")
    ax_b.set_axisbelow(True)

    bins = np.linspace(-1, 1, 41)
    ax_h.hist(pt_cur.clip(-1, 1), bins=bins, color="#9aa0a6", alpha=0.7, label=f"No route, no history (median {pt_cur.median():.2f})")
    ax_h.hist(pt_best.clip(-1, 1), bins=bins, color="#2a6ebb", alpha=0.55, label=f"GPS rasters + history (median {pt_best.median():.2f})")
    ax_h.set_xlabel("Per-truck R² over Sep–Oct (values below −1 shown at −1)")
    ax_h.set_ylabel("Trucks")
    ax_h.set_title(f"(b) Per-truck accuracy, {len(pt_best)} trucks", fontsize=11)
    ax_h.legend(frameon=False, loc="upper left", fontsize=9)
    ax_h.grid(color="#e5e5e5")
    ax_h.set_axisbelow(True)

    cur = res["No route,\nno history"]["pred"]
    for i, tid in enumerate(trucks):
        ax = fig.add_subplot(gs[i + 1, :])
        m = (test["TruckId"] == tid).to_numpy()
        tt = test[m]
        y, p, lo, hi = tt["energy"].to_numpy(), res[best]["pred"][m], res[best]["lo"][m], res[best]["hi"][m]
        cov = np.mean((y >= lo) & (y <= hi))
        ax.fill_between(tt["date"], lo, hi, color="#2a6ebb", alpha=0.15, lw=0, label="90% interval")
        ax.plot(tt["date"], y, color="black", lw=1.2, marker="o", ms=2.5, label="Actual")
        ax.plot(tt["date"], cur[m], color="#9aa0a6", lw=1.2, ls="--", label="No route, no history")
        ax.plot(tt["date"], p, color="#2a6ebb", lw=1.8, label="GPS rasters + history")
        ax.set_title(f"Truck {tid} — {tt['truck_category'].iloc[0]} | GPS rasters + history: R² {r2_score(y, p):.2f}, "
                     f"MAE {mean_absolute_error(y, p):.1f}, 90% cover {cov:.0%} | no route/history: R² {r2_score(y, cur[m]):.2f}",
                     fontsize=10)
        ax.set_ylabel("Daily energy")
        ax.set_ylim(0, max(y.max(), hi.max()) * 1.1)
        ax.set_xlim(test["date"].min(), test["date"].max())
        ax.grid(color="#e5e5e5")
        ax.set_axisbelow(True)
        if i == 0:
            ax.legend(frameon=False, loc="upper left", ncol=4, fontsize=9)

    ax = fig.add_subplot(gs[n + 1, :])
    ax.axis("off")
    tab = summary.copy()
    tab.columns = ["R² all", "MAE", "Median truck R²", "Trucks R²>0", "Trucks R²>0.5", "Fleet daily error"]
    cells = [[i] + [f"{v:.2f}" if j < 3 else f"{v:.0%}" if j < 5 else f"{v:.1%}" for j, v in enumerate(row)]
             for i, row in zip(tab.index, tab.to_numpy())]
    t = ax.table(cellText=cells, colLabels=["Route input"] + list(tab.columns), loc="center", cellLoc="center",
                 colWidths=[0.22] + [0.11] * 6)
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.8)

    fig.tight_layout()
    fig.savefig(HERE / "energy_forecast_comparison.png")
    plt.close(fig)
    print("per-truck R2 (GPS rasters + history):", pt_best.reindex(trucks).round(2).to_dict())


if __name__ == "__main__":
    main()
