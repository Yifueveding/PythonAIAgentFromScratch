"""Energy prediction when the truck's daily distance is not known exactly.

Scenarios (all trained Jan-Aug, tested Sep-Oct, same truck-day target as
predict_energy.py):
  actual distance : the truck's own GPS km that day (upper bound, from predict_energy.py)
  noisy distance  : planned km = actual km * exp(N(0, sigma)), i.e. a planning error
                    of roughly +/- sigma. Noise is applied in training as well, so the
                    model learns how much to trust the planned km.
  no distance     : no truck-day driving measures at all.

Moving GPS points and average speed are only observable after the day, so they
are dropped in every scenario here. All scenarios keep the truck attributes,
zone passes (as a stand-in for the known route), fleet distance, temperature
and month.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from predict_energy import CATEGORICAL, GB_PARAMS, build_dataset

HERE = Path(__file__).resolve().parent
SPLIT = "2023-09-01"
TRUCK = 155
BASE = ["age", "km_k", "in_fleet", "zone_passes", "distance_k", "temp", "month"]
SIGMAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
SHOW = {"no distance": None, "20% noise": 0.20, "10% noise": 0.10}  # scenarios drawn with intervals


def design(d: pd.DataFrame, numeric: list[str], columns=None) -> pd.DataFrame:
    X = pd.get_dummies(d[numeric + CATEGORICAL], columns=CATEGORICAL, dtype=float)
    return X if columns is None else X.reindex(columns=columns, fill_value=0.0)


def with_noise(d: pd.DataFrame, sigma: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return d.assign(planned_km=d["truck_km"] * np.exp(rng.normal(0.0, sigma, len(d))))


def run(train, test, sigma, quantiles=False):
    """Fit on train and predict test; sigma=None means no distance feature."""
    numeric = BASE if sigma is None else BASE + ["planned_km"]
    if sigma is not None:
        train, test = with_noise(train, sigma, seed=1), with_noise(test, sigma, seed=2)
    Xtr = design(train, numeric)
    Xte = design(test, numeric, Xtr.columns)
    out = {"pred": GradientBoostingRegressor(**GB_PARAMS).fit(Xtr, train["energy"]).predict(Xte)}
    if quantiles:
        for k, a in [("lo", 0.05), ("hi", 0.95)]:
            m = GradientBoostingRegressor(loss="quantile", alpha=a, **GB_PARAMS).fit(Xtr, train["energy"])
            out[k] = m.predict(Xte)
    return out


def metrics(test, p) -> dict:
    y = test["energy"].to_numpy()
    fleet = test.assign(pred=p["pred"]).groupby("date")[["energy", "pred"]].sum()
    r = {"R2": r2_score(y, p["pred"]), "MAE": mean_absolute_error(y, p["pred"]),
         "fleet MAPE": np.mean(np.abs(fleet["energy"] - fleet["pred"]) / fleet["energy"])}
    if "lo" in p:
        r["coverage"] = np.mean((y >= p["lo"]) & (y <= p["hi"]))
    return r


def main():
    d = build_dataset()
    train, test = d[d["date"] < SPLIT], d[d["date"] >= SPLIT]

    # --- accuracy vs planning error ----------------------------------------
    sweep = {}
    for s in SIGMAS:
        sweep[s] = metrics(test, run(train, test, s))
        print(f"sigma {s:.0%}: " + ", ".join(f"{k} {v:.3f}" for k, v in sweep[s].items()))
    no_dist = run(train, test, None, quantiles=True)
    nd = metrics(test, no_dist)
    print("no distance: " + ", ".join(f"{k} {v:.3f}" for k, v in nd.items()))

    # --- scenarios with intervals, for the single-truck panel ---------------
    shown = {"no distance": no_dist}
    for name, s in SHOW.items():
        if s is not None:
            shown[name] = run(train, test, s, quantiles=True)
            print(f"{name}: " + ", ".join(f"{k} {v:.3f}" for k, v in metrics(test, shown[name]).items()))

    # --- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(16, 13), dpi=170)
    gs = fig.add_gridspec(len(SHOW) + 1, 2, height_ratios=[1.2] + [1] * len(SHOW))
    ax_r2, ax_mae = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    xs = [s * 100 for s in SIGMAS]
    for ax, key, label in [(ax_r2, "R2", "R² (truck-day)"), (ax_mae, "MAE", "MAE per truck-day")]:
        ax.plot(xs, [sweep[s][key] for s in SIGMAS], color="#2a6ebb", lw=2, marker="o", label="Planned distance with error")
        ax.axhline(nd[key], color="#b8434f", lw=2, ls="--", label="No distance")
        for x, s in zip(xs, SIGMAS):
            ax.annotate(f"{sweep[s][key]:.2f}" if key == "R2" else f"{sweep[s][key]:.1f}", (x, sweep[s][key]),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
        ax.set_xlabel("Planning error in daily distance (σ, % of actual km)")
        ax.set_ylabel(label)
        ax.grid(color="#e5e5e5")
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="best")
    ax_r2.set_ylim(0, 1.05)
    ax_mae.set_ylim(bottom=0)
    ax_r2.set_title("(a) Accuracy vs distance planning error — all trucks, Sep–Oct test", fontsize=12)
    ax_mae.set_title(f"(b) Mean absolute error (fleet daily total error: "
                     f"{sweep[0.10]['fleet MAPE']:.1%} at 10% noise, {nd['fleet MAPE']:.1%} without distance)", fontsize=12)

    t_mask = (test["TruckId"] == TRUCK).to_numpy()
    tt = test[t_mask]
    for i, (name, p) in enumerate(shown.items()):
        ax = fig.add_subplot(gs[i + 1, :])
        lo, hi, pred = p["lo"][t_mask], p["hi"][t_mask], p["pred"][t_mask]
        y = tt["energy"].to_numpy()
        cov = np.mean((y >= lo) & (y <= hi))
        ax.fill_between(tt["date"], lo, hi, color="#2a6ebb", alpha=0.18, lw=0, label="90% interval")
        ax.plot(tt["date"], y, color="black", lw=1.2, marker="o", ms=2.5, label="Actual")
        ax.plot(tt["date"], pred, color="#2a6ebb", lw=1.8, label="Predicted")
        ax.set_title(f"({chr(ord('c') + i)}) Truck {TRUCK}, {name}: R² {r2_score(y, pred):.2f}, "
                     f"MAE {mean_absolute_error(y, pred):.1f}, 90% coverage {cov:.0%}", fontsize=11)
        ax.set_ylabel("Daily energy")
        ax.set_ylim(0, max(y.max(), hi.max()) * 1.1)
        ax.set_xlim(tt["date"].min(), tt["date"].max())
        ax.grid(color="#e5e5e5")
        ax.set_axisbelow(True)
        if i == 0:
            ax.legend(frameon=False, loc="upper left", ncol=3)

    fig.tight_layout()
    fig.savefig(HERE / "energy_prediction_distance_scenarios.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
